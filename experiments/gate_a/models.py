from __future__ import annotations

import torch
from torch import nn


class CausalBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int):
        super().__init__()
        pad = 2 * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=pad, dilation=dilation)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
        self.drop = nn.Dropout(0.1)

    def crop(self, x):
        return x[..., :-self.pad]

    def forward(self, x):
        y = self.drop(torch.nn.functional.gelu(self.crop(self.conv1(x))))
        y = self.drop(torch.nn.functional.gelu(self.crop(self.conv2(y))))
        return torch.nn.functional.gelu(y + self.skip(x))


class NodeTCN(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(CausalBlock(6, 64, 1), CausalBlock(64, width, 2),
                                 CausalBlock(width, width, 4))
        self.norm = nn.LayerNorm(width)

    def forward(self, history):
        b, t, n, _ = history.shape
        last = history[:, -1, :, :2]
        pos = (history[..., :2] - last[:, None]) / 20.0
        vel = history[..., 2:4] / 15.0
        acc = torch.diff(vel, dim=1, prepend=vel[:, :1]) / 0.1
        x = torch.cat([pos, vel, acc], -1).permute(0, 2, 3, 1).reshape(b * n, 6, t)
        return self.norm(self.net(x)[:, :, -1]).reshape(b, n, -1)


class GateAModel(nn.Module):
    def __init__(self, kind: str, max_nodes: int, width: int = 128, dt_s: float = 0.1):
        super().__init__()
        if kind not in {"self", "own", "pool", "graph", "all_graph"}:
            raise ValueError(kind)
        self.kind, self.max_nodes, self.width, self.dt_s = kind, max_nodes, width, dt_s
        self.encoder = NodeTCN(width)
        branch_in = width
        if kind == "own":
            self.own = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
            branch_in += width
        elif kind == "pool":
            self.pool = nn.Sequential(nn.Linear(width + 4, width), nn.GELU(), nn.Linear(width, width))
            branch_in += width
        elif kind in {"graph", "all_graph"}:
            self.message = nn.Sequential(nn.Linear(width * 2 + 4, width), nn.GELU(),
                                         nn.Linear(width, width), nn.GELU())
            self.attention = nn.Linear(width, 1)
            branch_in += width
        self.time = nn.Embedding(40, 24)
        self.decoder = nn.Sequential(nn.Linear(branch_in + 24 + 2, 192), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 2))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, history, node_mask):
        if self.kind == "graph":
            history, node_mask = history[:, :, :8], node_mask[:, :8]
        latent = self.encoder(history)
        target = latent[:, 0]
        current = history[:, -1]
        if self.kind == "own":
            target = torch.cat([target, self.own(target)], -1)
        elif self.kind == "pool":
            mask = node_mask[:, 1:].to(history.dtype)
            values = self.pool(torch.cat([latent[:, 1:], current[:, 1:]], -1))
            context = (values * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            target = torch.cat([target, context], -1)
        elif self.kind in {"graph", "all_graph"}:
            neighbors = latent[:, 1:]
            rel = current[:, 1:] - current[:, :1]
            own = latent[:, :1].expand(-1, neighbors.shape[1], -1)
            message = self.message(torch.cat([own, neighbors, rel], -1))
            mask = node_mask[:, 1:]
            score = self.attention(message).squeeze(-1).masked_fill(~mask, -1e4)
            weights = torch.softmax(score, 1) * mask
            context = (message * weights[..., None]).sum(1)
            target = torch.cat([target, context], -1)
        steps = torch.arange(40, device=history.device)
        times = (steps.to(history.dtype) + 1) * self.dt_s
        cv = history[:, -1, 0, 2:4][:, None] * times[None, :, None]
        x = torch.cat([target[:, None].expand(-1, 40, -1), self.time(steps)[None].expand(len(history), -1, -1),
                       cv / 20.0], -1)
        return cv + self.decoder(x) * 10.0
