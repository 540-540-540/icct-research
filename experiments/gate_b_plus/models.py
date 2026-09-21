from __future__ import annotations

import torch
from torch import nn

from experiments.gate_a.models import NodeTCN
from prediction.qgnn_paper_native.raj_paper import RajWeightedMultiJQGNNCore, RajWeightedSubsetQGNNCore
from prediction.qgnn_paper_native.raj_subset import MultiJFusion, RajMultiJJohnsonCore


class WideMultiJQGNNCore(nn.Module):
    """Higher-capacity D=8,k=4 quantum register; still the only cross-agent core."""

    def __init__(self, rounds: int):
        super().__init__()
        self.j2 = RajWeightedSubsetQGNNCore(j=2, rounds=rounds, D=8, k=4, encoder_hidden=128)
        self.j3 = RajWeightedSubsetQGNNCore(j=3, rounds=rounds, D=8, k=4, encoder_hidden=128)
        self.fuse = MultiJFusion()

    def forward(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.j2(history, mask), self.j3(history, mask), mask)


class GateBPlusQuantumModel(nn.Module):
    """Quantum-primary interaction model; no classical cross-agent message passing."""

    def __init__(self, mode: str, width: int = 128, dt_s: float = 0.1, rounds: int = 3,
                 core_kind: str = "quantum", branch_drop_probability: float = 0.0):
        super().__init__()
        if mode not in {"neighbor_residual", "dual_quantum_view", "full_multiscale", "multiscale_quantum_view",
                        "fused_multiscale", "cross_order_multiscale", "quantum_latent_attention",
                        "all_neighbor_quantum_latent_attention", "horizon_multiscale",
                        "horizon_independent", "horizon_trajectory", "horizon_kinematic",
                        "horizon_adaptive", "horizon_ranknorm", "multiscale_quantum_attention",
                        "target_conditioned_quantum_attention",
                        "stochastic_multiscale_quantum_attention",
                        "dual_readout_quantum_attention", "residual_multiscale_quantum_attention"}:
            raise ValueError(mode)
        if core_kind not in {"quantum", "quantum_wide", "johnson", "johnson_wide", "johnson_large"}:
            raise ValueError(core_kind)
        if not 0.0 <= branch_drop_probability < 1.0:
            raise ValueError(branch_drop_probability)
        self.mode, self.core_kind, self.dt_s = mode, core_kind, float(dt_s)
        self.branch_drop_probability = float(branch_drop_probability)
        self.encoder = NodeTCN(width)
        if core_kind == "quantum":
            self.core = RajWeightedMultiJQGNNCore(rounds=rounds)
        elif core_kind == "quantum_wide":
            self.core = WideMultiJQGNNCore(rounds=rounds)
        elif core_kind == "johnson":
            self.core = RajMultiJJohnsonCore(rounds=rounds, hidden=64)
        elif core_kind == "johnson_wide":
            self.core = RajMultiJJohnsonCore(rounds=rounds, hidden=70)
        else:
            self.core = RajMultiJJohnsonCore(rounds=rounds, hidden=128)
        core_width = {"neighbor_residual": 64, "dual_quantum_view": 128,
                      "full_multiscale": 128, "multiscale_quantum_view": 256,
                      "fused_multiscale": 192, "cross_order_multiscale": 256,
                      "quantum_latent_attention": 256,
                      "all_neighbor_quantum_latent_attention": 256,
                      "multiscale_quantum_attention": 256,
                      "target_conditioned_quantum_attention": 256,
                      "stochastic_multiscale_quantum_attention": 256,
                      "residual_multiscale_quantum_attention": 256,
                      "dual_readout_quantum_attention": 384,
                      "horizon_multiscale": 128, "horizon_independent": 128,
                      "horizon_trajectory": 128, "horizon_kinematic": 128,
                      "horizon_adaptive": 128, "horizon_ranknorm": 128}[mode]
        self.interaction = nn.Sequential(nn.Linear(core_width, width), nn.GELU(), nn.Linear(width, width))
        self.attention = (nn.Linear(width, 1)
                          if mode in {"quantum_latent_attention", "all_neighbor_quantum_latent_attention"}
                          else None)
        if mode in {"multiscale_quantum_attention", "target_conditioned_quantum_attention",
                    "stochastic_multiscale_quantum_attention",
                    "dual_readout_quantum_attention",
                    "residual_multiscale_quantum_attention"}:
            self.branch_messages = nn.ModuleList(
                nn.Sequential(nn.Linear(128, width), nn.GELU(), nn.Linear(width, width)) for _ in range(2)
            )
            self.branch_attentions = nn.ModuleList(nn.Linear(width, 1) for _ in range(2))
        else:
            self.branch_messages = self.branch_attentions = None
        if mode == "dual_readout_quantum_attention":
            self.shared_message = nn.Sequential(
                nn.Linear(256, width), nn.GELU(), nn.Linear(width, width)
            )
            self.shared_attention = nn.Linear(width, 1)
        else:
            self.shared_message = self.shared_attention = None
        if mode == "target_conditioned_quantum_attention":
            self.quantum_target_head = nn.Sequential(
                nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 2)
            )
            self.target_context = nn.Sequential(
                nn.Linear(2, 64), nn.GELU(), nn.Linear(64, width)
            )
            nn.init.zeros_(self.quantum_target_head[-1].weight)
            nn.init.zeros_(self.quantum_target_head[-1].bias)
        else:
            self.quantum_target_head = self.target_context = None
        if mode in {"horizon_multiscale", "horizon_independent", "horizon_trajectory",
                    "horizon_kinematic", "horizon_adaptive", "horizon_ranknorm"}:
            self.branch_norms = nn.ModuleList((nn.LayerNorm(64), nn.LayerNorm(64)))
            self.branch_projections = nn.ModuleList(
                nn.Sequential(nn.Linear(64, width), nn.GELU(), nn.Linear(width, width)) for _ in range(2)
            )
            self.horizon_gate = nn.Sequential(nn.Linear(24, 32), nn.SiLU(), nn.Linear(32, 2))
            nn.init.zeros_(self.horizon_gate[-1].weight)
            nn.init.zeros_(self.horizon_gate[-1].bias)
            scale_init = torch.tensor([0.10, 0.15])
            if mode == "horizon_independent":
                scale_init = torch.log(torch.expm1(scale_init))
            self.branch_scales = nn.Parameter(scale_init)
            if mode == "horizon_adaptive":
                self.horizon_query = nn.Linear(width + 24, 32)
                self.branch_keys = nn.ModuleList(nn.Linear(width, 32, bias=False) for _ in range(2))
                nn.init.normal_(self.horizon_query.weight, std=1e-3)
                nn.init.zeros_(self.horizon_query.bias)
            else:
                self.horizon_query = self.branch_keys = None
            self.branch_post_norms = (nn.ModuleList((nn.LayerNorm(width), nn.LayerNorm(width)))
                                      if mode == "horizon_ranknorm" else None)
        else:
            self.branch_norms = self.branch_projections = self.horizon_gate = None
            self.horizon_query = self.branch_keys = None
            self.branch_post_norms = None
            self.register_parameter("branch_scales", None)
        self.time = nn.Embedding(40, 24)
        self.decoder = nn.Sequential(
            nn.Linear(2 * width + 24 + 2, 192), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 2),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        if mode == "horizon_trajectory":
            self.trajectory_heads = nn.ModuleList(
                nn.Sequential(nn.Linear(width + 24, 64), nn.GELU(), nn.Linear(64, 2)) for _ in range(2)
            )
            for head in self.trajectory_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        else:
            self.trajectory_heads = None
        if mode == "horizon_kinematic":
            self.kinematic_head = nn.Sequential(nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, 4))
            nn.init.zeros_(self.kinematic_head[-1].weight)
            nn.init.zeros_(self.kinematic_head[-1].bias)
        else:
            self.kinematic_head = None

    def forward(self, history: torch.Tensor, node_mask: torch.Tensor,
                return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if history.ndim != 4 or history.shape[1:] != (20, history.shape[2], 4) or history.shape[2] < 8:
            raise ValueError("expected history [B,20,N>=8,4]")
        if self.mode != "all_neighbor_quantum_latent_attention":
            history, node_mask = history[:, :, :8], node_mask[:, :8]
        history = torch.where(node_mask[:, None, :, None], history, 0.0)
        own_mask = node_mask.clone()
        own_mask[:, 1:] = False
        if self.mode in {"full_multiscale", "multiscale_quantum_view", "fused_multiscale",
                         "cross_order_multiscale", "quantum_latent_attention",
                         "all_neighbor_quantum_latent_attention", "horizon_multiscale",
                         "horizon_independent", "horizon_trajectory", "horizon_kinematic",
                         "horizon_adaptive", "horizon_ranknorm", "multiscale_quantum_attention",
                         "target_conditioned_quantum_attention",
                         "stochastic_multiscale_quantum_attention",
                         "dual_readout_quantum_attention", "residual_multiscale_quantum_attention"}:
            j2_nodes = self.core.j2(history, node_mask)
            j3_nodes = self.core.j3(history, node_mask)
            j2_full = j2_nodes[:, 0]
            j3_full = j3_nodes[:, 0]
            if self.mode in {"quantum_latent_attention", "all_neighbor_quantum_latent_attention"}:
                quantum_nodes = torch.cat((j2_nodes, j3_nodes), -1)
                neighbors = quantum_nodes[:, 1:]
                target_quantum = quantum_nodes[:, :1].expand(-1, neighbors.shape[1], -1)
                messages = self.interaction(torch.cat((target_quantum, neighbors), -1))
                neighbor_mask = node_mask[:, 1:]
                scores = self.attention(messages).squeeze(-1).masked_fill(~neighbor_mask, -1e4)
                weights = torch.softmax(scores, 1) * neighbor_mask
                interaction_context = (messages * weights[..., None]).sum(1)
                quantum_context = None
            elif self.mode in {"multiscale_quantum_attention", "target_conditioned_quantum_attention",
                               "stochastic_multiscale_quantum_attention",
                               "dual_readout_quantum_attention",
                               "residual_multiscale_quantum_attention"}:
                neighbor_mask = node_mask[:, 1:]
                pooled = []
                branch_weights = []
                for nodes, message_layer, attention_layer in zip(
                        (j2_nodes, j3_nodes), self.branch_messages, self.branch_attentions):
                    neighbors = nodes[:, 1:]
                    target = nodes[:, :1].expand(-1, neighbors.shape[1], -1)
                    messages = message_layer(torch.cat((target, neighbors), -1))
                    scores = attention_layer(messages).squeeze(-1).masked_fill(~neighbor_mask, -1e4)
                    weights = torch.softmax(scores, 1) * neighbor_mask
                    branch_weights.append(weights)
                    attention_pool = (messages * weights[..., None]).sum(1)
                    if self.mode == "residual_multiscale_quantum_attention":
                        count = neighbor_mask.sum(1, keepdim=True).clamp_min(1)
                        mean_pool = (messages * neighbor_mask[..., None]).sum(1) / count
                        pooled.append(0.5 * (mean_pool + attention_pool))
                    else:
                        pooled.append(attention_pool)
                branch_pooled = tuple(pooled)
                if (self.mode == "stochastic_multiscale_quantum_attention" and self.training
                        and self.branch_drop_probability):
                    choice = torch.rand(len(history), device=history.device)
                    half = self.branch_drop_probability / 2.0
                    keep_j2 = (choice >= half).to(pooled[0].dtype)[:, None]
                    keep_j3 = ((choice < half) | (choice >= self.branch_drop_probability)).to(
                        pooled[1].dtype)[:, None]
                    scale = 1.0 / (1.0 - half)
                    pooled = [pooled[0] * keep_j2 * scale, pooled[1] * keep_j3 * scale]
                if self.mode == "dual_readout_quantum_attention":
                    quantum_nodes = torch.cat((j2_nodes, j3_nodes), -1)
                    neighbors = quantum_nodes[:, 1:]
                    target = quantum_nodes[:, :1].expand(-1, neighbors.shape[1], -1)
                    messages = self.shared_message(torch.cat((target, neighbors), -1))
                    scores = self.shared_attention(messages).squeeze(-1).masked_fill(~neighbor_mask, -1e4)
                    weights = torch.softmax(scores, 1) * neighbor_mask
                    pooled.insert(0, (messages * weights[..., None]).sum(1))
                quantum_context = torch.cat(pooled, -1)
            elif self.mode in {"full_multiscale", "horizon_multiscale", "horizon_independent",
                               "horizon_trajectory", "horizon_kinematic", "horizon_adaptive",
                               "horizon_ranknorm"}:
                quantum_context = torch.cat((j2_full, j3_full), -1)
            elif self.mode == "multiscale_quantum_view":
                j2_own = self.core.j2(history, own_mask)[:, 0]
                j3_own = self.core.j3(history, own_mask)[:, 0]
                quantum_context = torch.cat((j2_full, j2_full - j2_own, j3_full, j3_full - j3_own), -1)
            elif self.mode == "fused_multiscale":
                fused = self.core.fuse(j2_nodes, j3_nodes, node_mask)[:, 0]
                quantum_context = torch.cat((j2_full, j3_full, fused), -1)
            else:
                quantum_context = torch.cat((j2_full, j3_full, j2_full - j3_full, j2_full * j3_full), -1)
        else:
            full = self.core(history, node_mask)[:, 0]
            own = self.core(history, own_mask)[:, 0]
            delta = full - own
            quantum_context = delta if self.mode == "neighbor_residual" else torch.cat((full, delta), -1)
        if self.mode not in {"quantum_latent_attention", "all_neighbor_quantum_latent_attention"}:
            interaction_context = self.interaction(quantum_context)
        temporal = self.encoder(history)[:, 0]
        steps = torch.arange(40, device=history.device)
        times = (steps.to(history.dtype) + 1) * self.dt_s
        cv = history[:, -1, 0, 2:4][:, None] * times[None, :, None]
        time_embedding = self.time(steps)
        if self.mode in {"horizon_multiscale", "horizon_independent", "horizon_trajectory",
                         "horizon_kinematic", "horizon_adaptive", "horizon_ranknorm"}:
            q2 = self.branch_projections[0](self.branch_norms[0](j2_full))
            q3 = self.branch_projections[1](self.branch_norms[1](j3_full))
            if self.branch_post_norms is not None:
                q2 = self.branch_post_norms[0](q2)
                q3 = self.branch_post_norms[1](q3)
            if self.mode in {"horizon_multiscale", "horizon_trajectory", "horizon_kinematic",
                             "horizon_adaptive", "horizon_ranknorm"}:
                if self.mode == "horizon_adaptive":
                    query_input = torch.cat((temporal[:, None].expand(-1, 40, -1),
                                             time_embedding[None].expand(len(history), -1, -1)), -1)
                    query = self.horizon_query(query_input)
                    keys = torch.stack((self.branch_keys[0](q2), self.branch_keys[1](q3)), 1)
                    gate_logits = (self.horizon_gate(time_embedding)[None]
                                   + torch.einsum("btd,bjd->btj", query, keys) / (32 ** 0.5))
                    gates = torch.softmax(gate_logits, -1)
                else:
                    gates = torch.softmax(self.horizon_gate(time_embedding), -1)
                scales = self.branch_scales
            else:
                gates = 0.5 + torch.sigmoid(self.horizon_gate(time_embedding))
                scales = torch.nn.functional.softplus(self.branch_scales)
            if self.mode == "horizon_adaptive":
                quantum_residual = (gates[:, :, :1] * scales[0] * q2[:, None]
                                    + gates[:, :, 1:] * scales[1] * q3[:, None])
            else:
                quantum_residual = (gates[None, :, :1] * scales[0] * q2[:, None]
                                    + gates[None, :, 1:] * scales[1] * q3[:, None])
            interaction_by_time = interaction_context[:, None] + quantum_residual
            target_by_time = torch.cat((temporal[:, None].expand(-1, 40, -1), interaction_by_time), -1)
        else:
            if self.quantum_target_head is not None:
                target_delta = self.quantum_target_head(interaction_context) * 10.0
                quantum_target = cv[:, -1] + target_delta
                target_residual = self.target_context(target_delta / 20.0)
                ratio = (times / times[-1])[None, :, None]
                interaction_by_time = interaction_context[:, None] + ratio * target_residual[:, None]
                target_by_time = torch.cat((temporal[:, None].expand(-1, 40, -1),
                                            interaction_by_time), -1)
            else:
                target = torch.cat((temporal, interaction_context), -1)
                target_by_time = target[:, None].expand(-1, 40, -1)
        decoded = torch.cat((target_by_time,
                             time_embedding[None].expand(len(history), -1, -1), cv / 20.0), -1)
        prediction = cv + self.decoder(decoded) * 10.0
        if self.trajectory_heads is not None:
            time_by_batch = time_embedding[None].expand(len(history), -1, -1)
            direct2 = self.trajectory_heads[0](torch.cat((q2[:, None].expand(-1, 40, -1), time_by_batch), -1))
            direct3 = self.trajectory_heads[1](torch.cat((q3[:, None].expand(-1, 40, -1), time_by_batch), -1))
            direct = gates[None, :, :1] * direct2 + gates[None, :, 1:] * direct3
            prediction = prediction + direct * (times / times[-1])[None, :, None] * 10.0
        if self.kinematic_head is not None:
            delta_velocity, delta_acceleration = self.kinematic_head(torch.cat((q2, q3), -1)).chunk(2, -1)
            prediction = (prediction + delta_velocity[:, None] * times[None, :, None]
                          + 0.5 * delta_acceleration[:, None] * times.square()[None, :, None])
        if return_aux:
            latents = {"j2": j2_full, "j3": j3_full}
            if self.branch_attentions is not None:
                latents.update({"attention_j2": branch_weights[0], "attention_j3": branch_weights[1],
                                "pooled_j2": branch_pooled[0],
                                "pooled_j3": branch_pooled[1],
                                "neighbor_mask": neighbor_mask})
            if self.quantum_target_head is not None:
                latents["quantum_target"] = quantum_target
            if self.branch_projections is not None:
                latents.update({"q2": q2, "q3": q3})
            return prediction, latents
        return prediction

    def initialize_quantum_evolution_near_identity(self, scale: float = 0.01) -> None:
        """Use a fixed nonzero near-identity start for compound evolutions."""
        if self.core_kind not in {"quantum", "quantum_wide"}:
            raise ValueError("near-identity quantum initialization requires a quantum core")
        with torch.no_grad():
            for branch_no, branch in enumerate((self.core.j2, self.core.j3)):
                for layer_no, layer in enumerate(branch.embed_layers):
                    angles = torch.linspace(-scale, scale, layer.theta.numel(),
                                            device=layer.theta.device, dtype=layer.theta.dtype)
                    layer.theta.copy_(angles.roll(branch_no + layer_no))

    def parameter_audit(self) -> dict[str, int]:
        groups = {"temporal_encoder": self.encoder, f"{self.core_kind}_interaction_core": self.core,
                  "interaction_projection": self.interaction, "time_embedding": self.time,
                  "decoder": self.decoder}
        if self.attention is not None:
            groups["quantum_latent_attention"] = self.attention
        if self.branch_messages is not None:
            groups["quantum_branch_messages"] = self.branch_messages
            groups["quantum_branch_attentions"] = self.branch_attentions
        if self.shared_message is not None:
            groups["quantum_shared_message"] = self.shared_message
            groups["quantum_shared_attention"] = self.shared_attention
        if self.quantum_target_head is not None:
            groups["quantum_target_head"] = self.quantum_target_head
            groups["quantum_target_context"] = self.target_context
        if self.branch_norms is not None:
            groups["quantum_branch_norms"] = self.branch_norms
            groups["quantum_branch_projections"] = self.branch_projections
            groups["quantum_horizon_gate"] = self.horizon_gate
        if self.branch_post_norms is not None:
            groups["quantum_branch_post_norms"] = self.branch_post_norms
        if self.horizon_query is not None:
            groups["quantum_adaptive_query"] = self.horizon_query
            groups["quantum_adaptive_keys"] = self.branch_keys
        if self.trajectory_heads is not None:
            groups["quantum_trajectory_heads"] = self.trajectory_heads
        if self.kinematic_head is not None:
            groups["quantum_kinematic_head"] = self.kinematic_head
        result = {name: sum(p.numel() for p in module.parameters()) for name, module in groups.items()}
        if self.branch_scales is not None:
            result["quantum_branch_scales"] = self.branch_scales.numel()
        result["total"] = sum(p.numel() for p in self.parameters())
        return result
