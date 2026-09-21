"""Graph-conditioned, uncertainty-guided motion-token GPT-2 forecaster."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - remote environment includes peft
    LoraConfig = None
    get_peft_model = None

from motion_tokenizer_v7 import MotionTokenizerConfig, MotionTokenizerV7
from target_interaction_graph import ForecasterConfig, TargetInteractionGNN


def resolve_gpt2_source() -> str:
    here = Path(__file__).resolve().parent
    for candidate in (Path("gpt2"), here / "gpt2"):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Local GPT-2 weights were not found in the project directory.")


class MultiTargetGraphLLM(nn.Module):
    """Refine a frozen, validated GNN forecast with a shared motion-token GPT-2.

    The coordinate correction head is initialized to zero, so epoch 0 is exactly
    the validated GNN.  Any retained checkpoint therefore represents validation-
    measured improvement rather than an architectural assumption.
    """

    def __init__(
        self,
        graph_backbone: TargetInteractionGNN,
        config: ForecasterConfig,
        llm_layers: int = 4,
        lora_rank: int = 8,
        dropout: float = 0.10,
        correction_scale_m: float = 3.0,
        use_uncertainty: bool = True,
        use_soft_tokens: bool = True,
        use_graph_context: bool = True,
        fixed_temperature: float = 0.22,
        freeze_graph_backbone: bool = True,
    ):
        super().__init__()
        self.graph_backbone = graph_backbone
        self.config = config
        self.correction_scale_m = correction_scale_m
        self.use_uncertainty = use_uncertainty
        self.use_soft_tokens = use_soft_tokens
        self.use_graph_context = use_graph_context
        self.fixed_temperature = fixed_temperature
        self.freeze_graph_backbone = freeze_graph_backbone
        if self.freeze_graph_backbone:
            for parameter in self.graph_backbone.parameters():
                parameter.requires_grad = False

        tokenizer_config = MotionTokenizerConfig(dt=config.dt)
        self.motion_tokenizer = MotionTokenizerV7(tokenizer_config)
        self.vocab_size = self.motion_tokenizer.vocab_size

        source = resolve_gpt2_source()
        gpt_config = GPT2Config.from_pretrained(source, local_files_only=True)
        gpt_config.num_hidden_layers = llm_layers
        gpt_config.use_cache = False
        self.d_llm = int(gpt_config.n_embd)
        self.gpt2 = GPT2Model.from_pretrained(source, config=gpt_config, local_files_only=True)
        for parameter in self.gpt2.parameters():
            parameter.requires_grad = False

        self.motion_embedding = nn.Embedding(self.vocab_size, self.d_llm)
        with torch.no_grad():
            pretrained = self.gpt2.wte.weight.detach()
            indices = torch.linspace(0, pretrained.shape[0] - 1, self.vocab_size).round().long()
            self.motion_embedding.weight.copy_(0.20 * pretrained.index_select(0, indices))

        if get_peft_model is not None and LoraConfig is not None and lora_rank > 0:
            self.gpt2 = get_peft_model(
                self.gpt2,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules=["c_attn", "c_proj"],
                    lora_dropout=dropout,
                    bias="none",
                ),
            )

        hidden_dim = config.hidden_dim
        self.graph_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.d_llm), nn.GELU()
        )
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 2),
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.token_adapter = nn.Sequential(
            nn.LayerNorm(self.d_llm),
            nn.Linear(self.d_llm, self.d_llm),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_llm, self.d_llm),
        )
        self.future_queries = nn.Parameter(torch.randn(config.prediction_length, self.d_llm) * 0.02)
        self.token_head = nn.Sequential(nn.LayerNorm(self.d_llm), nn.Linear(self.d_llm, self.vocab_size))
        self.coordinate_head = nn.Sequential(
            nn.LayerNorm(self.d_llm + hidden_dim),
            nn.Linear(self.d_llm + hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, config.prediction_length * 2),
        )
        nn.init.zeros_(self.coordinate_head[-1].weight)
        nn.init.zeros_(self.coordinate_head[-1].bias)

    def _soft_history_embeddings(
        self,
        target_history: torch.Tensor,
        node_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        positions = target_history[..., :2]
        state_sequence = self.motion_tokenizer.build_state_sequence(positions, dt=self.config.dt)
        acceleration = torch.diff(target_history[..., 2:4], dim=1) / self.config.dt
        acceleration_scale = acceleration.norm(dim=-1).mean(dim=-1, keepdim=True).clamp_max(10.0) / 10.0
        speed_variation = target_history[..., 2:4].norm(dim=-1).std(dim=-1, keepdim=True) / 5.0
        uncertainty_input = torch.cat([node_features, acceleration_scale, speed_variation], dim=-1)
        # 0.04--0.40 m controls how broadly each observed transition is distributed
        # across nearby motion bins.
        if self.use_uncertainty:
            temperature = 0.04 + 0.36 * torch.sigmoid(self.uncertainty_head(uncertainty_input))
        else:
            temperature = torch.full(
                uncertainty_input.shape[:-1] + (1,),
                self.fixed_temperature,
                device=uncertainty_input.device,
                dtype=uncertainty_input.dtype,
            )
        logits = self.motion_tokenizer.motion_logits_from_state_sequence(state_sequence, temperature=temperature)
        probabilities = torch.softmax(logits, dim=-1)
        if not self.use_soft_tokens:
            hard_ids = probabilities.argmax(dim=-1)
            probabilities = F.one_hot(hard_ids, num_classes=self.vocab_size).to(probabilities.dtype)
        embedding_table = F.layer_norm(self.motion_embedding.weight, (self.d_llm,))
        embeddings = probabilities @ embedding_table
        return {"embeddings": embeddings, "temperature": temperature, "probabilities": probabilities}

    def forward(self, history: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, history_length, targets, _ = history.shape
        if self.freeze_graph_backbone:
            # Keep the accepted phase-1 predictor deterministic and unchanged.
            self.graph_backbone.eval()
            with torch.no_grad():
                graph_output = self.graph_backbone(history, target_mask)
        else:
            graph_output = self.graph_backbone(history, target_mask)
        nodes = graph_output["node_features"]
        target_history = history.permute(0, 2, 1, 3).reshape(batch * targets, history_length, 4)
        flat_nodes = nodes.reshape(batch * targets, -1)
        soft = self._soft_history_embeddings(target_history, flat_nodes)
        if self.use_graph_context:
            graph_token = self.graph_projection(flat_nodes).unsqueeze(1)
            coordinate_nodes = flat_nodes
        else:
            graph_token = torch.zeros(
                flat_nodes.shape[0], 1, self.d_llm, device=flat_nodes.device, dtype=flat_nodes.dtype
            )
            coordinate_nodes = torch.zeros_like(flat_nodes)
        motion_embeddings = soft["embeddings"] + graph_token
        sequence = torch.cat([graph_token, motion_embeddings + self.token_adapter(motion_embeddings)], dim=1)
        hidden = self.gpt2(inputs_embeds=sequence).last_hidden_state
        pooled = hidden[:, -1]

        future_hidden = pooled[:, None, :] + self.future_queries[None, :, :]
        token_logits = self.token_head(future_hidden).reshape(batch, targets, self.config.prediction_length, self.vocab_size)
        token_logits = token_logits.permute(0, 2, 1, 3)
        coordinate_input = torch.cat([pooled, coordinate_nodes], dim=-1)
        correction = self.coordinate_head(coordinate_input)
        correction = torch.tanh(correction.view(batch, targets, self.config.prediction_length, 2)) * self.correction_scale_m
        correction = correction.permute(0, 2, 1, 3) * target_mask[:, None, :, None]
        future_position = graph_output["future_position"] + correction
        return {
            "future_position": future_position,
            "displacement": future_position - history[:, -1, None, :, :2],
            "token_logits": token_logits,
            "token_temperature": soft["temperature"].view(batch, targets),
            "graph_future_position": graph_output["future_position"],
            "attention": graph_output["attention"],
            "adjacency": graph_output["adjacency"],
        }

    def future_token_ids(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Build one motion-token target for each future transition."""
        positions = torch.cat([history[:, -1:, :, :2], future[..., :2]], dim=1)
        batch, length, targets, _ = positions.shape
        target_major = positions.permute(0, 2, 1, 3).reshape(batch * targets, length, 2)
        ids = self.motion_tokenizer.motion_token_ids_from_positions(target_major)
        return ids.view(batch, targets, self.config.prediction_length).permute(0, 2, 1)

    def trainable_parameter_summary(self) -> Dict[str, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable": trainable, "total": total}
