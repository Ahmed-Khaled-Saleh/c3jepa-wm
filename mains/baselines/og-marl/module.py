"""
ViT-fronted adaptation of og_marl's IQLCQLSystem
(og_marl/baselines/torch_systems/offline/iql_cql.py), replacing DeepRNN's
flat-vector nn.Linear front-end with a ViT-Tiny encoder consuming raw
224x224x3 RGB observations, for use with FindGoalOfflineDataset +
OfflineReplayBuffer instead of FlashbaxReplayBuffer/Vault.

Design decisions, and why (fuller reasoning in chat):

1. ENCODE BEFORE AGENT-ID CONCAT. og_marl's own batch_concat_agent_id_to_obs
   actually already anticipates image observations -- it has a branch that
   appends an extra constant-valued "agent identity" channel directly onto
   a raw image ((H,W,C) -> (H,W,C+1)), rather than one-hot vector
   concatenation. That's a legitimate alternative design, but NOT what's
   used here: it would require modifying vit_hf's patch-embedding conv to
   accept 4 input channels instead of 3, adding real complexity/fragility
   for a from-scratch backbone. Instead: encode raw RGB (3-channel,
   unmodified from CommNet/MAPPO's encoder) into a flat hidden_dim feature
   vector FIRST, THEN call the SAME batch_concat_agent_id_to_obs utility,
   completely unmodified, on the now-flat (B,T,N,hidden_dim) features --
   which correctly triggers its OTHER (vector-obs, one-hot) branch. Net
   effect: identical agent-ID semantics to upstream's flat-vector-
   observation environments, just applied one step later in the pipeline.

2. BOTH online and target networks need their own encoder. Target
   Q-values now depend on encoded visual features too, not just the
   downstream GRU/linear head -- so target_encoder = deepcopy(encoder),
   synced on the same target_update_period as target_q_network. Both
   target_encoder and target_q_network are set to .eval() once at
   construction (upstream never toggles train/eval mode at all, but the
   frozen target path has no reason to run in train mode, and it avoids
   gradient-checkpointing overhead on a path that's always under
   torch.no_grad() anyway).

3. Do NOT eagerly cast observations to .float() before encoding (unlike
   upstream's `torch.from_numpy(experience["observations"]).float()`).
   The encoder's own preprocessing (torchvision v2 ToDtype(scale=True))
   detects the incoming dtype to decide how to rescale pixel values into
   [0,1] -- if already float32 by the time it arrives, that rescaling
   silently becomes a no-op (float32->float32 isn't rescaled), badly
   corrupting the input. Observations must stay uint8 until the encoder's
   own preprocessing runs.

4. terminals/truncations are PER-AGENT (B,T,N), not team-level (B,T) --
   see dataset.py and chat for why (og_marl's own reshape utilities
   require a 3-D+ input here; traced through FlashbaxReplayBuffer.add()
   to confirm upstream's inline "(B,T)" comments are stale). This is
   already what FindGoalOfflineDataset produces.

Everything else below (CQL loss, TD targets, double-Q action selection,
gather/switch/merge reshape helpers, unroll_rnn) is UNCHANGED from
upstream -- none of it cares whether the "obs_dim" features it operates on
came from a flat SMAC-style vector or a ViT. A shape-only numpy dry run of
this exact pipeline (encoder insertion + per-agent terminals) was run and
passed before writing this file -- see chat.

NOTE on evaluation-mode: upstream's IQLCQLSystem never calls .eval() on
q_network at all (train-mode is left active even during evaluate()
rollouts). This file matches that for the online q_network/encoder (not
introducing a new inconsistency), but does set the TARGET networks to
.eval() per point 2 above -- a narrower, clearly-scoped addition, not a
general train/eval-mode audit of the whole system.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')  # jax.tree.map below is the only jax usage in
                                                  # this file (a pytree-traversal convenience,
                                                  # no real computation) -- without this, just
                                                  # importing jax can eagerly reserve GPU memory
                                                  # via its default CUDA backend discovery, which
                                                  # then isn't available to torch. Must run before
                                                  # any jax API call forces backend init, hence
                                                  # right after the import, not deferred.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import tree
from chex import Numeric

import torchvision.transforms.v2 as v2

from og_marl.wrapped_environments.base import BaseEnvironment
from og_marl.loggers import BaseLogger
from og_marl.replay_buffers import Experience
from og_marl.baselines.base import BaseOfflineSystem
from og_marl.baselines.torch_systems.networks import DeepRNN
from og_marl.baselines.torch_systems.utils import (
    batch_concat_agent_id_to_obs,
    concat_agent_id_to_obs,
    expand_batch_and_agent_dim_of_time_major_sequence,
    gather,
    merge_batch_and_agent_dim_of_time_major_sequence,
    switch_two_leading_dims,
    unroll_rnn,
)

from stable_pretraining.backbone.utils import vit_hf


img_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VitEncoderModule(nn.Module):
    """
    (*batch, H, W, C) uint8 -> (*batch, hidden_dim) float feature.
    Same backbone/preprocessing as CommNetAgent.encode / MAPPO's
    VitEncoderModule -- duplicated here rather than imported, since this
    project doesn't yet have a shared module for it across the three
    training scripts (worth factoring out later if that gets annoying).
    """

    def __init__(self, hidden_dim: int, pretrained: bool = False,
                 gradient_checkpoint: bool = False):
        super().__init__()
        self.encoder = vit_hf(
            size="tiny", patch_size=14, image_size=224,
            pretrained=pretrained, use_mask_token=True,
        )
        self.processor = img_transform
        self.hidden_dim = hidden_dim

        self._manual_checkpoint = False
        if gradient_checkpoint:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()
            else:
                self._manual_checkpoint = True

    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        """obs_uint8: (*batch, H, W, C) uint8 -> (*batch, hidden_dim)."""
        *batch, H, W, C = obs_uint8.shape
        x = obs_uint8.reshape(-1, H, W, C).movedim(-1, -3)  # (prod(batch), 3, 224, 224)
        x = self.processor(x)  # MUST see uint8 input here -- see module docstring point 3

        if self._manual_checkpoint and self.training:
            def _encoder_fwd(inp):
                return self.encoder(inp)['last_hidden_state']
            hidden = torch.utils.checkpoint.checkpoint(_encoder_fwd, x, use_reentrant=False)
        else:
            hidden = self.encoder(x)['last_hidden_state']

        feats = hidden[:, 0]  # CLS token
        return feats.reshape(*batch, self.hidden_dim)


class ViTIQLCQLSystem(BaseOfflineSystem):
    """
    ViT-fronted IQL+CQL. See module docstring for the design decisions
    that differ from og_marl's original IQLCQLSystem. Everything else
    (CQL loss, double-Q targets, reshape helpers) is unchanged.
    """

    def __init__(
        self,
        environment: BaseEnvironment,
        logger: BaseLogger,
        hidden_dim: int = 192,
        pretrained_encoder: bool = False,
        gradient_checkpoint_encoder: bool = True,
        cql_weight: float = 2.0,
        linear_layer_dim: int = 64,
        recurrent_layer_dim: int = 64,
        discount: float = 0.99,
        target_update_period: int = 200,
        learning_rate: float = 3e-4,
        add_agent_id_to_obs: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        save_checkpoints: bool = True,
    ):
        super().__init__(environment, logger)

        self.device = torch.device(device)
        self.discount = discount
        self.add_agent_id_to_obs = add_agent_id_to_obs
        self.num_agents = len(self.environment.agents)

        # Encoder: raw RGB -> hidden_dim feature, per-agent per-timestep.
        # Always 3-channel in -- see module docstring point 1 for why
        # agent-id conditioning is NOT baked into the encoder itself.
        self.encoder = VitEncoderModule(
            hidden_dim, pretrained=pretrained_encoder,
            gradient_checkpoint=gradient_checkpoint_encoder,
        ).to(self.device)

        q_input_dim = hidden_dim
        if self.add_agent_id_to_obs:
            q_input_dim += self.num_agents  # one-hot ID, concatenated POST-encoding

        self.q_network = DeepRNN(
            input_dim=q_input_dim,
            linear_layer_dim=linear_layer_dim,
            recurrent_layer_dim=recurrent_layer_dim,
            output_dim=self.environment.num_actions,
        ).to(self.device)

        # Target networks -- BOTH the encoder and the Q-head need target
        # copies (module docstring point 2), synced together.
        self.target_encoder = copy.deepcopy(self.encoder).to(self.device)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_encoder.eval()
        self.target_q_network.eval()
        self.target_update_period = target_update_period

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.q_network.parameters()),
            lr=learning_rate,
        )

        self.rnn_states = {
            agent: self.q_network.initial_state(1, self.device)
            for agent in self.environment.agents
        }

        self.cql_weight = cql_weight

        # ---- checkpointing ----
        # Hyperparameters needed to reconstruct this system's architecture
        # later (before calling load_checkpoint) -- stored so a checkpoint
        # is self-describing rather than requiring you to remember/hardcode
        # them separately, same convention as train.py/train_mappo.py's
        # save_checkpoint helpers elsewhere in this project.
        self._init_kwargs = {
            "hidden_dim": hidden_dim,
            "pretrained_encoder": pretrained_encoder,
            "gradient_checkpoint_encoder": gradient_checkpoint_encoder,
            "cql_weight": cql_weight,
            "linear_layer_dim": linear_layer_dim,
            "recurrent_layer_dim": recurrent_layer_dim,
            "discount": discount,
            "target_update_period": target_update_period,
            "learning_rate": learning_rate,
            "add_agent_id_to_obs": add_agent_id_to_obs,
        }
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_checkpoints = save_checkpoints
        self.best_eval_score = float("-inf")

    def reset(self) -> None:
        """Called at the start of a new episode during evaluation."""
        self.rnn_states = {
            agent: self.q_network.initial_state(1, self.device)
            for agent in self.environment.agents
        }

    def save_checkpoint(self, path, extra_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves everything needed to resume training or run evaluation:
        encoder/q_network/target_encoder/target_q_network state dicts,
        optimizer state, training_step_ctr, and the constructor
        hyperparameters needed to rebuild this system's architecture
        before calling load_checkpoint (see __init__'s self._init_kwargs).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "training_step_ctr": self.training_step_ctr,
            "encoder_state_dict": self.encoder.state_dict(),
            "q_network_state_dict": self.q_network.state_dict(),
            "target_encoder_state_dict": self.target_encoder.state_dict(),
            "target_q_network_state_dict": self.target_q_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "init_kwargs": self._init_kwargs,
            "extra_info": extra_info or {},
        }, path)

    def load_checkpoint(self, path, map_location=None) -> Dict[str, Any]:
        """
        Loads a checkpoint saved by save_checkpoint into this ALREADY
        CONSTRUCTED system -- build the ViTIQLCQLSystem with matching
        hyperparameters first (checkpoint["init_kwargs"] tells you what
        they were, if you saved it and lost track), then call this.
        Returns the checkpoint's "extra_info" dict, e.g.:

            ckpt = torch.load(path, map_location="cpu")
            print(ckpt["init_kwargs"])  # sanity check before rebuilding
            system = ViTIQLCQLSystem(env, logger, **ckpt["init_kwargs"])
            extra_info = system.load_checkpoint(path)
        """
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.q_network.load_state_dict(ckpt["q_network_state_dict"])
        self.target_encoder.load_state_dict(ckpt["target_encoder_state_dict"])
        self.target_q_network.load_state_dict(ckpt["target_q_network_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.training_step_ctr = ckpt["training_step_ctr"]
        return ckpt.get("extra_info", {})

    def evaluate(self, num_eval_episodes: int = 32) -> Dict[str, Numeric]:
        """
        Overrides BaseOfflineSystem.evaluate() to additionally report
        success rate -- all agents individually reached the goal, none
        truncated -- alongside the original reward-based metrics, plus
        piggybacks periodic checkpoint saving onto this call (train()
        already calls evaluate() at evaluation_every-step intervals and
        once at the end, and has no separate checkpoint hook of its own --
        see chat).

        Reward alone can't tell you this: _reward() fires per INDIVIDUAL
        agent success event, not gated on joint/team success (traced
        through FindGoalEnv.on_success/handle_actions in chat) -- a
        nonzero return only means "at least one individual success
        happened somewhere," and with joint_reward=True (this project's
        default) even ONE agent succeeding pays the WHOLE team, making
        return an even weaker proxy for "did every agent reach the goal."
        Requires an explicit per-agent terminal/truncation check, same
        approach as CommNet's/MAPPO's evaluate() functions built earlier
        in this project.
        """
        episode_returns = []
        episode_lengths = []
        successes = []
        any_goal_reached = []

        for _ in range(num_eval_episodes):
            self.reset()
            observations, infos = self.environment.reset()

            done = False
            episode_return = 0.0
            episode_length = 0
            ever_terminated = {agent: False for agent in self.agents}
            ever_truncated = {agent: False for agent in self.agents}

            while not done:
                legal_actions = infos["legals"] if "legals" in infos else None

                actions = self.select_actions(observations, legal_actions)
                observations, rewards, terminal, truncation, infos = self.environment.step(actions)

                episode_return += np.mean(list(rewards.values()), dtype="float")
                episode_length += 1

                for agent in self.agents:
                    ever_terminated[agent] = ever_terminated[agent] or bool(terminal[agent])
                    ever_truncated[agent] = ever_truncated[agent] or bool(truncation[agent])

                done = all(terminal.values()) or all(truncation.values())

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            successes.append(all(ever_terminated.values()) and not any(ever_truncated.values()))
            any_goal_reached.append(any(ever_terminated.values()))

        logs = {
            "evaluation/mean_episode_return": float(np.mean(episode_returns)),
            "evaluation/max_episode_return": float(np.max(episode_returns)),
            "evaluation/min_episode_return": float(np.min(episode_returns)),
            "evaluation/mean_episode_length": float(np.mean(episode_lengths)),
            "evaluation/success_rate": float(np.mean(successes)),
            "evaluation/any_goal_reached_rate": float(np.mean(any_goal_reached)),
        }

        if self.save_checkpoints:
            self.save_checkpoint(self.checkpoint_dir / "last.pt")
            self.save_checkpoint(self.checkpoint_dir / f"step_{self.training_step_ctr}.pt")

            # composite score, not raw success_rate: comparing "0.0 > 0.0"
            # forever after the first eval call would freeze "best.pt" on
            # a near-random early checkpoint the moment success_rate hits
            # a genuine (possibly long) run of zeros -- see the identical
            # bug found and fixed in train.py's best_eval.pt earlier in
            # this project.
            eval_score = logs["evaluation/success_rate"] + 0.01 * logs["evaluation/any_goal_reached_rate"]
            if eval_score >= self.best_eval_score:
                self.best_eval_score = eval_score
                self.save_checkpoint(self.checkpoint_dir / "best.pt")

        return logs
        self.rnn_states = {
            agent: self.q_network.initial_state(1, self.device)
            for agent in self.environment.agents
        }

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        legal_actions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        # Keep raw pixel dtype (uint8) here too -- do NOT .float() before
        # encoding, same reasoning as _train_step (module docstring point 3).
        observations = tree.map_structure(
            lambda x: torch.from_numpy(x).to(self.device), observations
        )
        legal_actions = tree.map_structure(
            lambda x: torch.from_numpy(x).bool().to(self.device), legal_actions
        )

        actions, next_rnn_states = self._select_actions(
            observations, legal_actions, self.rnn_states
        )
        self.rnn_states = next_rnn_states

        return tree.map_structure(lambda x: x.cpu().numpy(), actions)

    def _select_actions(
        self,
        observations: Dict[str, torch.Tensor],
        legal_actions: Dict[str, torch.Tensor],
        rnn_states: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        actions = {}
        next_rnn_states = {}

        with torch.no_grad():
            for i, agent in enumerate(self.agents):
                agent_observation = observations[agent]  # (H, W, 3) uint8, single step

                # VitEncoderModule expects (*batch, H, W, C); add a
                # singleton batch dim to encode this one observation, then
                # squeeze it back off.
                encoded = self.encoder(agent_observation.unsqueeze(0))  # (1, hidden_dim)
                encoded = encoded.squeeze(0)                             # (hidden_dim,)

                if self.add_agent_id_to_obs:
                    encoded = concat_agent_id_to_obs(encoded, i, len(self.agents))

                encoded = encoded.unsqueeze(0)  # add batch dimension for the GRU
                q_values, next_rnn_states[agent] = self.q_network(
                    encoded, rnn_states[agent]
                )

                agent_legal_actions = legal_actions[agent]
                masked_q_values = torch.where(
                    agent_legal_actions,
                    q_values[0],
                    torch.tensor(-99999999.0, device=self.device),
                )
                greedy_action = torch.argmax(masked_q_values)
                actions[agent] = greedy_action

        return actions, next_rnn_states

    def train_step(self, experience: Experience) -> Dict[str, Numeric]:
        experience = jax.tree.map(lambda x: np.array(x), experience)
        logs = self._train_step(experience)
        return {k: v.item() if torch.is_tensor(v) else v for k, v in logs.items()}

    def _train_step(self, experience: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # Keep observations as their original uint8 dtype -- do NOT
        # .float() here (module docstring point 3); the encoder's own
        # preprocessing needs to see uint8 to rescale correctly.
        observations = torch.from_numpy(experience["observations"]).to(self.device)             # (B,T,N,H,W,C) uint8
        actions = torch.from_numpy(experience["actions"]).long().to(self.device)                 # (B,T,N)
        rewards = torch.from_numpy(experience["rewards"]).float().to(self.device)                 # (B,T,N)
        truncations = torch.from_numpy(experience["truncations"]).float().to(self.device)         # (B,T,N) per-agent
        terminals = torch.from_numpy(experience["terminals"]).float().to(self.device)             # (B,T,N) per-agent
        legal_actions = torch.from_numpy(experience["infos"]["legals"]).bool().to(self.device)     # (B,T,N,A)

        resets = torch.maximum(terminals, truncations).bool()  # (B,T,N)

        B, T, N, A = legal_actions.shape

        # ---- encode raw pixels -> flat features (online + target) ----
        online_encoded = self.encoder(observations)              # (B,T,N,hidden_dim)
        with torch.no_grad():
            target_encoded = self.target_encoder(observations)   # (B,T,N,hidden_dim)

        # ---- agent-id concat, POST-encoding (module docstring point 1) ----
        if self.add_agent_id_to_obs:
            online_encoded = batch_concat_agent_id_to_obs(online_encoded)
            target_encoded = batch_concat_agent_id_to_obs(target_encoded)

        # ---- everything below is UNCHANGED from upstream ----
        online_encoded = switch_two_leading_dims(online_encoded)
        target_encoded = switch_two_leading_dims(target_encoded)
        resets = switch_two_leading_dims(resets)

        online_encoded = merge_batch_and_agent_dim_of_time_major_sequence(online_encoded)
        target_encoded = merge_batch_and_agent_dim_of_time_major_sequence(target_encoded)
        resets = merge_batch_and_agent_dim_of_time_major_sequence(resets)

        # Unroll target network (encoder-through-Q-head, both frozen here)
        with torch.no_grad():
            target_qs_out = unroll_rnn(self.target_q_network, target_encoded, resets)

        target_qs_out = expand_batch_and_agent_dim_of_time_major_sequence(target_qs_out, B, N)
        target_qs_out = switch_two_leading_dims(target_qs_out)

        self.optimizer.zero_grad()

        # Unroll online network
        qs_out = unroll_rnn(self.q_network, online_encoded, resets)

        qs_out = expand_batch_and_agent_dim_of_time_major_sequence(qs_out, B, N)
        qs_out = switch_two_leading_dims(qs_out)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qs = gather(qs_out, actions, dim=3)

        # Max over target Q-Values / Double Q-learning
        qs_out_selector = torch.where(
            legal_actions, qs_out, torch.tensor(-9999999.0, device=self.device)
        )
        cur_max_actions = torch.argmax(qs_out_selector, dim=3)
        target_max_qs = gather(target_qs_out, cur_max_actions, dim=-1)

        # Compute targets (per-agent terminals now broadcast correctly
        # against per-agent rewards/target_max_qs, elementwise)
        targets = (
            rewards[:, :-1] + (1 - terminals[:, :-1]) * self.discount * target_max_qs[:, 1:]
        )
        targets = targets.detach()

        # TD-Error Loss
        td_loss = F.mse_loss(chosen_action_qs[:, :-1], targets)

        #############
        #### CQL ####
        #############
        cql_loss = torch.mean(
            torch.logsumexp(qs_out, dim=-1, keepdim=True)[:, :-1]
        ) - torch.mean(chosen_action_qs[:, :-1])
        #############
        #### end ####
        #############

        loss = td_loss + self.cql_weight * cql_loss

        loss.backward()
        self.optimizer.step()

        # Maybe update target networks (BOTH encoder and Q-head)
        if self.training_step_ctr % self.target_update_period == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_encoder.load_state_dict(self.encoder.state_dict())

        return {
            "loss": loss,
            "cql_loss": cql_loss,
            "td_loss": td_loss,
            "mean_q_values": torch.mean(qs_out),
            "mean_chosen_q_values": torch.mean(chosen_action_qs),
        }