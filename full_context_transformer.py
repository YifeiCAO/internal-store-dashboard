from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FullContextTransformerConfig:
    dim_action: int = 6
    dim_sensory: int = 384
    dim_model: int = 80
    n_heads: int = 8
    n_layers: int = 2
    dim_feedforward: int = 216
    max_sequence_length: int = 384
    attention_window: int | None = None
    use_write_flag: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.dim_action <= 0 or self.dim_sensory <= 0:
            raise ValueError("input dimensions must be positive")
        if self.dim_model <= 0 or self.n_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.dim_model % self.n_heads:
            raise ValueError("dim_model must be divisible by n_heads")
        if (self.dim_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.n_layers <= 0 or self.dim_feedforward <= 0:
            raise ValueError("network depth and feedforward size must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.attention_window is not None and self.attention_window <= 0:
            raise ValueError("attention_window must be positive or None")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


@dataclass
class FullContextTransformerOutput:
    prediction: torch.Tensor
    hidden: torch.Tensor
    future_ground_truth_reads: int = 0
    future_ground_truth_writes: int = 0


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return (value.float() * scale).to(value.dtype) * self.weight


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dimension: int, max_sequence_length: int) -> None:
        super().__init__()
        frequencies = 1.0 / (
            10_000.0
            ** (
                torch.arange(0, head_dimension, 2, dtype=torch.float32)
                / head_dimension
            )
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.shape[-2]
        cosine = self.cosine[:sequence_length].to(
            device=query.device,
            dtype=query.dtype,
        )
        sine = self.sine[:sequence_length].to(
            device=query.device,
            dtype=query.dtype,
        )
        cosine = cosine.view(1, 1, sequence_length, -1)
        sine = sine.view(1, 1, sequence_length, -1)
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )


def build_causal_attention_mask(
    sequence_length: int,
    *,
    window: int | None,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=device)
    distance = positions[:, None] - positions[None, :]
    allowed = distance >= 0
    if window is not None:
        allowed &= distance < window
    mask = torch.zeros(
        sequence_length,
        sequence_length,
        device=device,
        dtype=torch.float32,
    )
    return mask.masked_fill(~allowed, -torch.inf)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: FullContextTransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dimension = config.dim_model // config.n_heads
        self.attention_window = config.attention_window
        self.query_key_value = nn.Linear(
            config.dim_model,
            3 * config.dim_model,
            bias=False,
        )
        self.output = nn.Linear(config.dim_model, config.dim_model, bias=False)
        self.dropout = config.dropout
        self.rotary = RotaryEmbedding(
            self.head_dimension,
            config.max_sequence_length,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, dimension = hidden.shape
        query_key_value = self.query_key_value(hidden)
        query, key, value = query_key_value.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dimension,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        query, key = self.rotary(query, key)
        mask = build_causal_attention_mask(
            sequence_length,
            window=self.attention_window,
            device=hidden.device,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            dimension,
        )
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(dimension, 2 * hidden_dimension, bias=False)
        self.output = nn.Linear(hidden_dimension, dimension, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = self.input(hidden).chunk(2, dim=-1)
        return self.output(self.dropout(F.silu(gate) * value))


class FullContextTransformerBlock(nn.Module):
    def __init__(self, config: FullContextTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim_model)
        self.attention = CausalSelfAttention(config)
        self.feedforward_norm = RMSNorm(config.dim_model)
        self.feedforward = SwiGLU(
            config.dim_model,
            config.dim_feedforward,
            config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.dropout(
            self.attention(self.attention_norm(hidden))
        )
        return hidden + self.dropout(
            self.feedforward(self.feedforward_norm(hidden))
        )


class FullContextTransformer(nn.Module):
    """Causal all-history Transformer for the waypoint recall protocol.

    The model consumes only action, frozen sensory features, and optionally the
    externally supplied write event used by ReMAP-Former. It never receives
    context, room, pose, place, target identity, query mask, or source indices.
    """

    model_kind = "full_context_transformer"

    def __init__(
        self,
        config: FullContextTransformerConfig = (
            FullContextTransformerConfig()
        ),
    ) -> None:
        super().__init__()
        self.config = config
        self.sensory_norm = nn.LayerNorm(
            config.dim_sensory,
            elementwise_affine=False,
        )
        self.sensory_encoder = nn.Linear(
            config.dim_sensory,
            config.dim_model,
            bias=False,
        )
        self.action_encoder = nn.Linear(
            config.dim_action,
            config.dim_model,
            bias=False,
        )
        self.write_encoder = (
            nn.Linear(1, config.dim_model, bias=False)
            if config.use_write_flag
            else None
        )
        self.input_norm = RMSNorm(config.dim_model)
        self.layers = nn.ModuleList(
            FullContextTransformerBlock(config)
            for _ in range(config.n_layers)
        )
        self.output_norm = RMSNorm(config.dim_model)
        self.decoder = nn.Linear(
            config.dim_model,
            config.dim_sensory,
            bias=False,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )
        residual_scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for block in self.layers:
            block.attention.output.weight.data.mul_(residual_scale)
            block.feedforward.output.weight.data.mul_(residual_scale)

    def forward(
        self,
        actions: torch.Tensor,
        sensory: torch.Tensor,
        *,
        write_mask: torch.Tensor | None = None,
    ) -> FullContextTransformerOutput:
        if actions.ndim != 3 or sensory.ndim != 3:
            raise ValueError("actions and sensory must have shape [B,T,D]")
        if actions.shape[:2] != sensory.shape[:2]:
            raise ValueError("action and sensory sequence shapes do not align")
        if actions.shape[-1] != self.config.dim_action:
            raise ValueError("action dimension does not match config")
        if sensory.shape[-1] != self.config.dim_sensory:
            raise ValueError("sensory dimension does not match config")
        if actions.shape[1] > self.config.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")

        hidden = self.sensory_encoder(self.sensory_norm(sensory))
        hidden = hidden + self.action_encoder(actions)
        if self.write_encoder is not None:
            if write_mask is None:
                raise ValueError("write_mask is required by this configuration")
            if write_mask.ndim == 2:
                write_mask = write_mask.unsqueeze(-1)
            if write_mask.shape != (*actions.shape[:2], 1):
                raise ValueError("write_mask must have shape [B,T] or [B,T,1]")
            hidden = hidden + self.write_encoder(
                write_mask.to(dtype=hidden.dtype)
            )
        hidden = self.input_norm(hidden)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.output_norm(hidden)
        return FullContextTransformerOutput(
            prediction=self.decoder(hidden),
            hidden=hidden,
        )

