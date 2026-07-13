
import torch
import torchvision.transforms.v2 as v2
import torch.nn.functional as F
import torch.nn as nn
from stable_pretraining.backbone.utils import vit_hf

from model import CommNetActorCritic


img_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


# --------------------------------------------------------------------------
# Model wrapper: encoder + CommNet in one module for convenience.
# --------------------------------------------------------------------------
class CommNetAgent(nn.Module):
    def __init__(self, hidden_dim: int, num_actions: int, num_comm_steps: int,
                 tie_weights: bool, gradient_checkpoint_encoder: bool = False):
        super().__init__()
        self.encoder = vit_hf(
                            size="tiny",
                            patch_size=14,
                            image_size=224,
                            pretrained=False,
                            use_mask_token=True,
                        )
        self.processor = img_transform

        # Gradient checkpointing: recomputes backbone activations during the
        # backward pass instead of storing them during forward, trading
        # extra compute for much lower activation memory -- unlike freezing,
        # this keeps full gradient flow into the backbone, so it's safe to
        # use while training the encoder from scratch. Prefer the encoder's
        # own (usually more efficient, HF-style) implementation if it has
        # one; otherwise fall back to wrapping the forward call manually.
        self._manual_checkpoint = False
        if gradient_checkpoint_encoder:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()
            else:
                self._manual_checkpoint = True

        self.commnet = CommNetActorCritic(
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_comm_steps=num_comm_steps,
            tie_weights=tie_weights,
        )

    def encode(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        """obs_uint8: (B, N, 224, 224, 3) -> (B, N, hidden_dim)"""
        B, N = obs_uint8.shape[:2]
        x = obs_uint8.movedim(-1, -3) # since we pass a tensor to transform, we need to move the channel dimension to the front manually
        x = self.processor(x)          # (B, N, 3, 224, 224)
        x = x.reshape(B * N, *x.shape[2:])                  # encoder applied per-agent

        if self._manual_checkpoint and self.training:
            # checkpoint() needs a function returning tensor(s), not a
            # dict-like model output, hence this thin wrapper extracting
            # just last_hidden_state before checkpointing.
            def _encoder_fwd(inp):
                return self.encoder(inp)['last_hidden_state']
            hidden = torch.utils.checkpoint.checkpoint(_encoder_fwd, x, use_reentrant=False)
        else:
            hidden = self.encoder(x)['last_hidden_state']   # (B*N, seq_len, hidden_dim)

        feats = hidden[:, 0]                                 # CLS token, (B*N, hidden_dim)
        return feats.reshape(B, N, -1)

    def forward(self, obs_uint8: torch.Tensor, mask=None):
        h0 = self.encode(obs_uint8)
        return self.commnet(h0, mask)

    @torch.no_grad()
    def act(self, obs_uint8: torch.Tensor, mask=None, deterministic=False):
        h0 = self.encode(obs_uint8)
        return self.commnet.act(h0, mask, deterministic)

