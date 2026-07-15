

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from stable_pretraining.backbone.utils import vit_hf

from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import MultiAgentMLP, ProbabilisticActor


# --------------------------------------------------------------------------
# Same image encoder as CommNet's CommNetAgent.encode -- reused verbatim.
# --------------------------------------------------------------------------
img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




class VitEncoderModule(nn.Module):
    """
    (*batch, n_agents, 224, 224, 3) uint8 -> (*batch, n_agents, hidden_dim)
    float feature per agent. Same backbone/preprocessing as
    `CommNetAgent.encode` in train.py, factored out standalone so it can
    sit inside a `TensorDictModule` for both the MAPPO actor and critic.

    `noise_std`: standard deviation of simple additive white Gaussian
    noise applied to the normalized image tensor before the backbone.
    0.0 (default) = no noise. This is used to simulate a noisy
    communication channel corrupting what the *centralized critic*
    receives (see `evaluate()`'s channel_noise_std) -- it's a plain
    instance attribute rather than a constructor-only setting so
    `evaluate()` can toggle it on/off around specific forward passes
    without rebuilding the module.

    `gradient_checkpoint`: trades compute for memory by recomputing
    backbone activations during backward instead of storing them, while
    still allowing full gradient flow -- important here since both the
    actor's and critic's encoders train from scratch (`pretrained=False`
    by default) and so can never be frozen; freezing would mean they never
    learn anything. With two independent ViT instances (actor + critic,
    intentionally not weight-shared -- see the note above `build_networks`),
    this matters at least as much here as in CommNet's single-encoder setup.
    """

    def __init__(self, hidden_dim: int, pretrained: bool = False, noise_std: float = 0.0,
                 gradient_checkpoint: bool = False):
        super().__init__()
        self.encoder = vit_hf(
            size="tiny", patch_size=14, image_size=224,
            pretrained=pretrained, use_mask_token=True,
        )
        self.processor = img_transform
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std

        self._manual_checkpoint = False
        if gradient_checkpoint:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()
            else:
                self._manual_checkpoint = True

    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        *batch, n_agents, H, W, C = obs_uint8.shape
        x = obs_uint8.reshape(-1, H, W, C).movedim(-1, -3)  # (*batch*n_agents, 3, 224, 224)
        x = self.processor(x)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std  # simple AWGN "channel effect"

        if self._manual_checkpoint and self.training:
            def _encoder_fwd(inp):
                return self.encoder(inp)['last_hidden_state']
            hidden = torch.utils.checkpoint.checkpoint(_encoder_fwd, x, use_reentrant=False)
        else:
            hidden = self.encoder(x)['last_hidden_state']

        feats = hidden[:, 0]  # CLS token, (*batch*n_agents, hidden)
        return feats.reshape(*batch, n_agents, self.hidden_dim)




GROUP = "agent"

def _keys(group: str = GROUP):
    return {
        "obs": (group, "observation", "pov"),
        "action": (group, "action"),
        "reward": (group, "reward"),
        "done": (group, "done"),
        "terminated": (group, "terminated"),
        "truncated": (group, "truncated"),
        "hidden": (group, "hidden"),
        "logits": (group, "logits"),
        "value": (group, "state_value"),
        "episode_reward": (group, "episode_reward"),
    }

# --------------------------------------------------------------------------
# Networks: shared VitEncoderModule design note.
# --------------------------------------------------------------------------
# It's tempting to save compute by giving the actor and critic the SAME
# VitEncoderModule instance (one ViT forward pass feeding both heads,
# exactly how CommNetAgent shares its encoder between policy and value
# heads). Don't do this with ClipPPOLoss's default `functional=True`:
# torchrl "functionalizes" actor_network and critic_network SEPARATELY
# internally, which silently breaks weight sharing between them even if
# you passed the same nn.Module object into both TensorDictSequentials --
# you'd end up training two independently-diverging copies of what you
# thought was one shared encoder. If you want a shared trunk, you must
# pass `ClipPPOLoss(..., functional=False)` and verify parameter identity
# yourself; simpler and safer default here is two separate encoder
# instances (2x ViT compute, but correct and matches the tutorial's
# fully-independent actor/critic networks).
def build_networks(cfg, n_agents: int, n_actions: int,
                    action_spec, device: torch.device):
    K = _keys()

    actor_encoder = TensorDictModule(
        VitEncoderModule(cfg.hidden_dim, cfg.pretrained_encoder,
                         gradient_checkpoint=cfg.gradient_checkpoint_encoder).to(device),
        in_keys=[K["obs"]], out_keys=[K["hidden"]],
    )
    actor_head = TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=cfg.hidden_dim, n_agent_outputs=n_actions, n_agents=n_agents,
            centralized=False, share_params=True, device=device,
            depth=cfg.actor_depth, num_cells=cfg.actor_num_cells, activation_class=nn.Tanh,
        ),
        in_keys=[K["hidden"]], out_keys=[K["logits"]],
    )
    policy_module = TensorDictSequential(actor_encoder, actor_head)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=[K["logits"]],
        out_keys=[K["action"]],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )

    critic_encoder = TensorDictModule(
        VitEncoderModule(cfg.hidden_dim, cfg.pretrained_encoder,  # separate instance -- see note above
                         gradient_checkpoint=cfg.gradient_checkpoint_encoder).to(device),
        in_keys=[K["obs"]], out_keys=[K["hidden"]],
    )
    critic_head = TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=cfg.hidden_dim, n_agent_outputs=1, n_agents=n_agents,
            centralized=True,  # MAPPO: critic conditions on every agent's encoded feature
            share_params=True, device=device,
            depth=cfg.critic_depth, num_cells=cfg.critic_num_cells, activation_class=nn.Tanh,
        ),
        in_keys=[K["hidden"]], out_keys=[K["value"]],
    )
    value_module = TensorDictSequential(critic_encoder, critic_head)

    return policy, value_module