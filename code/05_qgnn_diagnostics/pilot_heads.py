"""Paired, frozen-feature pilot heads; validation is V_select, never test."""
import json
import math
from pathlib import Path

import torch
from torch import nn


ARMS = ("own_only", "PV24", "VP24", "PV48_square", "PV48_relation")


class PilotHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.own = nn.Linear(6, 128)
        self.graph = nn.Linear(48, 128)
        layer = nn.TransformerEncoderLayer(
            128, 4, 256, dropout=0.0, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 40))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        p = torch.arange(20, dtype=torch.float32)[:, None]
        f = torch.exp(torch.arange(0, 128, 2) * (-math.log(10000.0) / 128))
        position = torch.zeros(20, 128)
        position[:, 0::2], position[:, 1::2] = torch.sin(p * f), torch.cos(p * f)
        self.register_buffer("position", position)

    def forward(self, state, standardized, exists, detected, graph):
        # Inputs are eligible vehicle histories [vehicles, time, channels].
        own = torch.cat((standardized, exists[..., None].float(),
                         detected[..., None].float()), -1)
        own = own.masked_fill(~exists[..., None], 0)
        graph = graph.masked_fill(~exists[..., None], 0)
        h = self.own(own) + self.graph(graph) + self.position
        n, t = exists.shape
        causal = torch.ones(t, t, dtype=torch.bool, device=h.device).triu(1)
        mask = causal[None].expand(n, -1, -1) | ~exists[:, None, :]
        # Invalid queries may attend themselves to avoid all-masked softmax;
        # valid queries still cannot attend any invalid key.
        diagonal = torch.eye(t, dtype=torch.bool, device=h.device)[None]
        mask = mask & ~(diagonal & ~exists[:, :, None])
        h = self.temporal(h, mask=mask.repeat_interleave(4, dim=0))
        last = torch.where(exists, torch.arange(t, device=h.device), -1).max(1).values
        rows = torch.arange(n, device=h.device)
        residual = self.output(h[rows, last]).reshape(n, 20, 2)
        current = state[rows, last]
        dt = torch.arange(1, 21, device=h.device, dtype=h.dtype)[None, :, None] * 0.1
        return current[:, None, :2] + current[:, None, 2:] * dt + residual


def _check(data):
    s = len(data["state_hat"])
    for name in ("state_hat", "standardized_state"):
        assert data[name].shape == (s, 20, 8, 4), (name, data[name].shape)
    for name in ("track_exists", "detected", "label_valid"):
        assert data[name].shape == (s, 20, 8), (name, data[name].shape)
        assert data[name].dtype == torch.bool, name
    assert data["future_position"].shape == (s, 20, 8, 2)
    for arm in ARMS:
        assert data["features"][arm].shape == (s, 20, 8, 48), arm
    assert s > 0


def _normalization(train, arm):
    if arm == "own_only":
        return torch.zeros(48), torch.ones(48)
    x = train["features"][arm][train["track_exists"]].float()
    if not len(x) or not torch.isfinite(x).all():
        raise ValueError(f"No valid finite training features: {arm}")
    return x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)


def _batch(data, indices, arm, mean, std, device):
    exists = data["track_exists"][indices].transpose(1, 2)
    eligible = exists[:, :, -1] & (exists.sum(-1) >= 3)
    scene, vehicle = eligible.nonzero(as_tuple=True)
    def take(name):
        return data[name][indices].transpose(1, 2)[scene, vehicle].to(device)
    raw = data["features"][arm][indices].transpose(1, 2)[scene, vehicle].float()
    graph = torch.zeros_like(raw) if arm == "own_only" else (raw - mean) / std
    inputs = [take(k).float() for k in ("state_hat", "standardized_state")]
    inputs += [take(k) for k in ("track_exists", "detected")]
    inputs += [graph.to(device)]
    return inputs, take("future_position").float(), take("label_valid"), scene.to(device), indices[scene], vehicle


def _errors(prediction, target, valid):
    # Invalid labels may be NaN; replace before arithmetic, not afterwards.
    safe_target = torch.where(valid[..., None], target, prediction.detach())
    distance = torch.linalg.vector_norm(prediction - safe_target, dim=-1)
    count = valid.sum(-1)
    ade = (distance * valid).sum(-1) / count.clamp_min(1)
    return ade, distance[:, -1], count, valid[:, -1]


def _scene_loss(ade, fde, count, final_valid, scene):
    losses = []
    for sid in scene.unique():
        use = (scene == sid) & (count > 0)
        if not use.any():
            continue
        final = (scene == sid) & final_valid
        loss = (ade[use] * count[use]).sum() / count[use].sum()
        if final.any():
            loss = loss + 0.5 * fde[final].mean()
        losses.append(loss)
    return torch.stack(losses).mean() if losses else None


@torch.no_grad()
def _evaluate(model, data, arm, mean, std, device):
    model.eval()
    records = []
    for start in range(0, len(data["state_hat"]), 8):
        ids = torch.arange(start, min(start + 8, len(data["state_hat"])))
        inputs, target, valid, _, sample, vehicle = _batch(data, ids, arm, mean, std, device)
        if not len(sample):
            continue
        ade, fde, count, final = _errors(model(*inputs), target, valid)
        for i in range(len(sample)):
            records.append({"sample_index": int(sample[i]), "vehicle_index": int(vehicle[i]),
                            "valid_steps": int(count[i]), "fde_valid": bool(final[i]),
                            "ADE": float(ade[i]) if count[i] else None,
                            "FDE": float(fde[i]) if final[i] else None})
    a = [r["ADE"] for r in records if r["ADE"] is not None]
    f = [r["FDE"] for r in records if r["FDE"] is not None]
    if not a or not f:
        raise ValueError("Evaluation needs eligible vehicles with ADE and true-final FDE labels")
    ade, fde = sum(a) / len(a), sum(f) / len(f)
    if not math.isfinite(ade + fde):
        raise ValueError("Nonfinite pilot metrics")
    return {"ADE": ade, "FDE": fde, "J": ade + 0.5 * fde,
            "ade_count": len(a), "fde_count": len(f), "records": records}


def run_heads(train, val, seed, outdir, device):
    """Train five matched heads; select checkpoints solely by V_select J."""
    _check(train)
    _check(val)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": int(seed), "selection_split": "V_select", "units": "m",
               "epochs_max": 12, "patience": 3, "batch_scenes": 8,
               "learning_rate": 1e-3, "weight_decay": 1e-2, "arms": {}}
    for arm in ARMS:
        torch.manual_seed(seed)
        model = PilotHead().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        generator = torch.Generator().manual_seed(seed)
        mean, std = _normalization(train, arm)
        best, best_epoch, stale, history = float("inf"), 0, 0, []
        checkpoint = outdir / f"head_seed{seed}_{arm}.pt"
        for epoch in range(13):
            if epoch:
                model.train()
                order = torch.randperm(len(train["state_hat"]), generator=generator)
                for indices in order.split(8):
                    inputs, target, valid, scene, sample, _ = _batch(
                        train, indices, arm, mean, std, device)
                    if not len(sample):
                        continue
                    loss = _scene_loss(*_errors(model(*inputs), target, valid), scene)
                    if loss is None:
                        continue
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite loss: {arm} epoch {epoch}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            tr = _evaluate(model, train, arm, mean, std, device)
            va = _evaluate(model, val, arm, mean, std, device)
            history.append({"epoch": epoch,
                            "train": {k: v for k, v in tr.items() if k != "records"},
                            "V_select": {k: v for k, v in va.items() if k != "records"}})
            if va["J"] < best:
                best, best_epoch, stale = va["J"], epoch, 0
                best_train, best_val = tr, va
                torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                            "seed": seed, "arm": arm, "epoch": epoch,
                            "selection_split": "V_select"}, checkpoint)
            elif epoch:
                stale += 1
            if stale >= 3:
                break
        summary["arms"][arm] = {
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "best_epoch": best_epoch, "history": history, "train": best_train,
            "V_select": best_val, "checkpoint": str(checkpoint),
            "normalization": {"mean": mean.tolist(), "std": std.tolist(),
                              "fitted_on": "train track_exists records"}}
        (outdir / f"heads_seed{seed}.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(f"seed={seed} arm={arm} best_epoch={best_epoch} "
              f"V_select ADE={best_val['ADE']:.6f} FDE={best_val['FDE']:.6f}", flush=True)
    return summary
