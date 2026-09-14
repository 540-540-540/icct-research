"""
V7 agent-centric motion tokenizer.

The tokenizer follows the key idea borrowed from MotionLM / Verlet-Agent:
tokenize local motion in the agent-centric frame instead of global scene-centric
coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class MotionTokenizerConfig:
    forward_bins: int = 21
    lateral_bins: int = 21
    forward_min: float = -0.5
    forward_max: float = 1.5
    lateral_min: float = -1.0
    lateral_max: float = 1.0
    dt: float = 0.05


class MotionTokenizerV7:
    def __init__(self, config: MotionTokenizerConfig | None = None):
        self.config = config or MotionTokenizerConfig()
        self.vocab_size = self.config.forward_bins * self.config.lateral_bins
        self.start_token_id = self.vocab_size

    def _centers(self, device, dtype):
        f = torch.linspace(
            self.config.forward_min,
            self.config.forward_max,
            steps=self.config.forward_bins,
            device=device,
            dtype=dtype,
        )
        l = torch.linspace(
            self.config.lateral_min,
            self.config.lateral_max,
            steps=self.config.lateral_bins,
            device=device,
            dtype=dtype,
        )
        return f, l

    def bin_centers(self, device, dtype):
        return self._centers(device, dtype)

    def _token_tables(self, device, dtype):
        f_centers, l_centers = self._centers(device, dtype)
        f_grid = f_centers.view(-1, 1).expand(-1, self.config.lateral_bins)
        l_grid = l_centers.view(1, -1).expand(self.config.forward_bins, -1)
        return f_grid.reshape(-1), l_grid.reshape(-1)

    @staticmethod
    def build_state_sequence(traj_real: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        """
        traj_real: [B, T, 2]
        return: [B, T, 5] = [x, y, vx, vy, heading]
        """
        vel = torch.zeros_like(traj_real)
        if traj_real.shape[1] >= 2:
            vel[:, 1:, :] = (traj_real[:, 1:, :] - traj_real[:, :-1, :]) / dt
            vel[:, 0, :] = vel[:, 1, :]

        heading = torch.atan2(vel[..., 1], vel[..., 0])
        if traj_real.shape[1] >= 2:
            stationary = vel.norm(dim=-1) < 1e-4
            heading[:, 0] = heading[:, 1]
            for t in range(1, traj_real.shape[1]):
                heading[:, t] = torch.where(stationary[:, t], heading[:, t - 1], heading[:, t])

        return torch.cat([traj_real, vel, heading.unsqueeze(-1)], dim=-1)

    def local_deltas_from_state_sequence(self, state_seq: torch.Tensor):
        """
        state_seq: [B, T, 5]
        return forward/lateral deltas for transitions t->t+1, shape [B, T-1]
        """
        pos = state_seq[..., :2]
        heading = state_seq[..., 4]
        delta = pos[:, 1:, :] - pos[:, :-1, :]
        h = heading[:, :-1]
        cos_h = torch.cos(h)
        sin_h = torch.sin(h)
        forward = delta[..., 0] * cos_h + delta[..., 1] * sin_h
        lateral = -delta[..., 0] * sin_h + delta[..., 1] * cos_h
        return forward, lateral

    def motion_token_ids_from_state_sequence(self, state_seq: torch.Tensor) -> torch.Tensor:
        forward, lateral = self.local_deltas_from_state_sequence(state_seq)
        f_centers, l_centers = self._centers(forward.device, forward.dtype)
        f_idx = self._continuous_to_index(forward, f_centers)
        l_idx = self._continuous_to_index(lateral, l_centers)
        return f_idx * self.config.lateral_bins + l_idx

    def motion_factor_bins_from_state_sequence(self, state_seq: torch.Tensor):
        forward, lateral = self.local_deltas_from_state_sequence(state_seq)
        f_centers, l_centers = self._centers(forward.device, forward.dtype)
        f_idx = self._continuous_to_index(forward, f_centers)
        l_idx = self._continuous_to_index(lateral, l_centers)
        return f_idx, l_idx

    def motion_token_ids_from_positions(self, traj_real: torch.Tensor) -> torch.Tensor:
        state_seq = self.build_state_sequence(traj_real, dt=self.config.dt)
        return self.motion_token_ids_from_state_sequence(state_seq)

    def motion_logits_from_state_sequence(self, state_seq: torch.Tensor, temperature: float | torch.Tensor = 0.12) -> torch.Tensor:
        forward, lateral = self.local_deltas_from_state_sequence(state_seq)
        token_f, token_l = self._token_tables(forward.device, forward.dtype)
        if not torch.is_tensor(temperature):
            temperature = torch.tensor(float(temperature), device=forward.device, dtype=forward.dtype)
        temperature = temperature.to(device=forward.device, dtype=forward.dtype).clamp_min(1e-4)
        while temperature.ndim < forward.ndim:
            temperature = temperature.unsqueeze(-1)
        diff_f = (forward.unsqueeze(-1) - token_f.view(1, 1, -1)) / temperature.unsqueeze(-1)
        diff_l = (lateral.unsqueeze(-1) - token_l.view(1, 1, -1)) / temperature.unsqueeze(-1)
        return -(diff_f ** 2 + diff_l ** 2)

    def motion_soft_targets_from_state_sequence(self, state_seq: torch.Tensor, temperature: float | torch.Tensor = 0.18) -> torch.Tensor:
        logits = self.motion_logits_from_state_sequence(state_seq, temperature=temperature)
        return torch.softmax(logits, dim=-1)

    def soft_decode(
        self,
        start_pos: torch.Tensor,
        start_heading: torch.Tensor,
        token_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        start_pos: [B, 2]
        start_heading: [B]
        token_probs: [B, H, V]
        return future positions [B, H, 2]
        """
        token_f, token_l = self._token_tables(token_probs.device, token_probs.dtype)
        forward = torch.sum(token_probs * token_f.view(1, 1, -1), dim=-1)
        lateral = torch.sum(token_probs * token_l.view(1, 1, -1), dim=-1)
        return self._rollout_local_motion(start_pos, start_heading, forward, lateral)

    def hard_decode(
        self,
        start_pos: torch.Tensor,
        start_heading: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_f, token_l = self._token_tables(start_pos.device, start_pos.dtype)
        forward = token_f[token_ids]
        lateral = token_l[token_ids]
        return self._rollout_local_motion(start_pos, start_heading, forward, lateral)

    def expected_local_motion_from_probs(self, token_probs: torch.Tensor):
        token_f, token_l = self._token_tables(token_probs.device, token_probs.dtype)
        forward = torch.sum(token_probs * token_f.view(1, 1, -1), dim=-1)
        lateral = torch.sum(token_probs * token_l.view(1, 1, -1), dim=-1)
        return forward, lateral

    def token_bin_coords(self, token_ids: torch.Tensor):
        f_idx = token_ids // self.config.lateral_bins
        l_idx = token_ids % self.config.lateral_bins
        return f_idx, l_idx

    def marginal_probs_from_joint(self, joint_probs: torch.Tensor):
        shape = joint_probs.shape[:-1] + (self.config.forward_bins, self.config.lateral_bins)
        joint = joint_probs.reshape(shape)
        forward_probs = joint.sum(dim=-1)
        lateral_probs = joint.sum(dim=-2)
        return forward_probs, lateral_probs

    def _rollout_local_motion(
        self,
        start_pos: torch.Tensor,
        start_heading: torch.Tensor,
        forward: torch.Tensor,
        lateral: torch.Tensor,
    ) -> torch.Tensor:
        pos = start_pos
        heading = start_heading
        traj = []
        eps = 1e-4
        for t in range(forward.shape[1]):
            f = forward[:, t]
            l = lateral[:, t]
            dx = torch.cos(heading) * f - torch.sin(heading) * l
            dy = torch.sin(heading) * f + torch.cos(heading) * l
            pos = pos + torch.stack([dx, dy], dim=-1)
            traj.append(pos)
            heading = heading + torch.atan2(l, torch.clamp(f.abs(), min=eps))
        return torch.stack(traj, dim=1)

    @staticmethod
    def _continuous_to_index(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(values.unsqueeze(-1) - centers.view(*([1] * values.ndim), -1))
        return torch.argmin(diff, dim=-1).long()
