"""Common numerical token, frozen GPT-2/LoRA and estimated-origin trajectory contract."""
from pathlib import Path
import math
import torch
from torch import nn
from transformers import GPT2Model


class NumericTokenAdapter(nn.Module):
    def __init__(self, graph_dim):
        super().__init__()
        if graph_dim not in (21, 24, 128):
            raise ValueError('Graph dimension must be 21, 24 or 128')
        self.state_projection = nn.Linear(4, 768)
        self.marker_projection = nn.Linear(2, 768, bias=False)
        self.graph_projection = nn.Linear(graph_dim, 768, bias=False)
        if graph_dim == 24:
            with torch.no_grad():
                self.graph_projection.weight[:, 21:].zero_()

    def forward(self, standardized_state, track_exists, detected, graph_features):
        b, t, n, c = standardized_state.shape
        if c != 4 or graph_features.shape != (b, t, n, self.graph_projection.in_features):
            raise ValueError('Expected matching [B,T,N,4] states and graph features')
        m = track_exists.bool()
        dtype = self.state_projection.weight.dtype
        z = torch.where(m[..., None], standardized_state, 0).to(dtype)
        q = torch.where(m[..., None], graph_features, 0).to(dtype)
        markers = torch.stack((m, detected.bool()), -1).to(dtype)
        tokens = self.state_projection(z) + self.marker_projection(markers) + self.graph_projection(q)
        tokens = tokens * m[..., None]
        # Position is added by GPT-2 only; no cross-vehicle attention.
        return tokens.permute(0, 2, 1, 3).reshape(b*n, t, 768)


class FusedQKVLoRA(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.lora_A = nn.Parameter(base.weight.new_empty(base.weight.shape[0], rank))
        self.lora_B = nn.Parameter(base.weight.new_zeros(rank, base.weight.shape[1]))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)


class TrajectoryPredictor(nn.Module):
    def __init__(self, graph_dim, checkpoint=None):
        super().__init__()
        checkpoint = checkpoint or Path(__file__).resolve().parents[1] / 'models/gpt2'
        self.adapter = NumericTokenAdapter(graph_dim)
        self.llm = GPT2Model.from_pretrained(str(checkpoint), local_files_only=True, attn_implementation='eager')
        if (self.llm.config.n_layer, self.llm.config.n_head, self.llm.config.n_embd) != (12, 12, 768):
            raise ValueError('Expected GPT-2 base 12/12/768 checkpoint')
        self.llm.requires_grad_(False)
        self.llm.config.use_cache = False
        for block in self.llm.h:
            block.attn.c_attn = FusedQKVLoRA(block.attn.c_attn)
        self.head = nn.Sequential(nn.Linear(768, 256), nn.SiLU(), nn.Linear(256, 40))

    def forward(self, state_hat, standardized_state, track_exists, detected, graph_features):
        if state_hat.shape != standardized_state.shape or state_hat.ndim != 4:
            raise ValueError('Expected matching [B,20,N,4] state tensors')
        b, t, n, _ = state_hat.shape
        if t != 20 or not 1 <= n <= 8:
            raise ValueError('History must span twenty time-grid points and at most eight slots')
        if track_exists.shape != state_hat.shape[:-1] or detected.shape != track_exists.shape:
            raise ValueError('Mask shape mismatch')
        m = track_exists.bool()
        eligible = m[:, -1] & (m.sum(1) >= 3)
        tokens = self.adapter(standardized_state, m, detected, graph_features)
        sequence_mask = m.permute(0, 2, 1).reshape(b*n, t)
        active = eligible.reshape(-1).nonzero(as_tuple=True)[0]
        # Keep an autograd zero for a wholly unavailable batch; no GPT-2 call.
        correction = tokens.new_zeros((b*n, 20, 2)) + tokens.sum() * 0
        if active.numel():
            output = self.llm(inputs_embeds=tokens[active], attention_mask=sequence_mask[active],
                              position_ids=torch.arange(t, device=tokens.device)[None, :],
                              use_cache=False).last_hidden_state
            last = (sequence_mask[active] * torch.arange(t, device=tokens.device)).max(-1).values
            hidden = output[torch.arange(len(active), device=tokens.device), last]
            correction = correction.index_copy(0, active, self.head(hidden).reshape(-1, 20, 2))
        correction = correction.reshape(b, n, 20, 2).permute(0, 2, 1, 3)
        origin = state_hat[:, -1].to(correction.dtype)
        horizon = torch.arange(1, 21, device=origin.device, dtype=origin.dtype)[None, :, None, None] * .1
        baseline = origin[:, None, :, :2] + horizon * origin[:, None, :, 2:]
        prediction = torch.where(eligible[:, None, :, None], baseline + correction, 0)
        return dict(prediction=prediction, origin_eligible=eligible)


def masked_trajectory_loss(prediction, future_position, label_valid, origin_eligible):
    """Scene-normalized training ADE + 0.5 FDE; unavailable scenes contribute zero.

    This is a training denominator, not the protocol's evaluation macro average.
    Callers must skip optimizer.step when skip_optimizer is true.
    """
    if prediction.shape != future_position.shape or prediction.ndim != 4 or prediction.shape[1] != 20 or prediction.shape[-1] != 2:
        raise ValueError('Expected matching [B,20,N,2] prediction and target')
    if label_valid.shape != prediction.shape[:-1] or origin_eligible.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError('Supervision mask shape mismatch')
    valid = label_valid.bool() & origin_eligible[:, None, :].bool()
    difference = torch.where(valid[..., None], prediction - future_position, 0)
    distance = torch.linalg.vector_norm(difference, dim=-1)
    ade_count = valid.sum((1, 2))
    fde_count = valid[:, -1].sum(1)
    ade = distance.sum((1, 2)) / ade_count.clamp_min(1)
    fde = distance[:, -1].sum(1) / fde_count.clamp_min(1)
    scene_loss = ade + .5 * fde
    return dict(loss=scene_loss.mean(), scene_ade=ade, scene_fde=fde,
                scene_loss=scene_loss, ade_count=ade_count, fde_count=fde_count,
                scenes_without_fde=(fde_count == 0).sum(),
                scenes_without_supervision=(ade_count == 0).sum(),
                skip_optimizer=bool((ade_count == 0).all()))
