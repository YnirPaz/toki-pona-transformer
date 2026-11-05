"""Decoder-only transformer used for Toki Pona language modeling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


MODEL_SIZES = ("small", "medium", "medium-do0.1", "medium-do0.1-v2")


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int
    context_size: int = 256
    embedding_dimension: int = 96
    layer_count: int = 4
    head_count: int = 4
    feed_forward_dimension: int = 384
    rms_norm_epsilon: float = 1e-5
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        if self.embedding_dimension % self.head_count != 0:
            raise ValueError("embedding_dimension must be divisible by head_count")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be at least 0 and less than 1")

    @property
    def head_dimension(self) -> int:
        return self.embedding_dimension // self.head_count


def model_config_for_size(
    model_size: str,
    vocabulary_size: int,
    *,
    context_size: int = 256,
) -> ModelConfig:
    """Build one of the named model configurations."""
    if model_size == "small":
        return ModelConfig(
            vocabulary_size=vocabulary_size,
            context_size=context_size,
            embedding_dimension=96,
            layer_count=4,
            head_count=4,
            feed_forward_dimension=384,
        )
    if model_size in ("medium", "medium-do0.1", "medium-do0.1-v2"):
        return ModelConfig(
            vocabulary_size=vocabulary_size,
            context_size=context_size,
            embedding_dimension=128,
            layer_count=4,
            head_count=4,
            feed_forward_dimension=512,
            dropout_probability=(
                0.1 if model_size in ("medium-do0.1", "medium-do0.1-v2") else 0.0
            ),
        )
    choices = ", ".join(MODEL_SIZES)
    raise ValueError(f"unknown model size {model_size!r}; choose one of: {choices}")


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        scale = torch.rsqrt(inputs.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return inputs * scale * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.head_count = config.head_count
        self.head_dimension = config.head_dimension
        self.dropout_probability = config.dropout_probability
        dimension = config.embedding_dimension
        self.query_key_value = nn.Linear(dimension, 3 * dimension)
        self.output = nn.Linear(dimension, dimension)
        self.output_dropout = (
            nn.Dropout(config.dropout_probability)
            if config.dropout_probability > 0.0
            else nn.Identity()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, dimension = inputs.shape
        query, key, value = self.query_key_value(inputs).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.head_count,
                self.head_dimension,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_probability if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            dimension,
        )
        return self.output_dropout(self.output(attended))

    def forward_with_cache(
        self,
        inputs: Tensor,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Attend to ``inputs`` and return keys/values for later decoding."""
        batch_size, sequence_length, dimension = inputs.shape
        query, key, value = self.query_key_value(inputs).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.head_count,
                self.head_dimension,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_length = past_key.shape[-2]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        attention_mask = None
        is_causal = past_key_value is None
        if past_key_value is not None and sequence_length > 1:
            key_positions = torch.arange(key.shape[-2], device=inputs.device)
            query_positions = past_length + torch.arange(
                sequence_length,
                device=inputs.device,
            )
            attention_mask = key_positions[None, :] <= query_positions[:, None]

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout_probability if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            dimension,
        )
        return self.output_dropout(self.output(attended)), (key, value)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(
            config.embedding_dimension,
            config.feed_forward_dimension,
        )
        self.output = nn.Linear(
            config.feed_forward_dimension,
            config.embedding_dimension,
        )
        self.output_dropout = (
            nn.Dropout(config.dropout_probability)
            if config.dropout_probability > 0.0
            else nn.Identity()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output_dropout(self.output(F.gelu(self.input(inputs))))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm attention and feed-forward residual block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(
            config.embedding_dimension,
            config.rms_norm_epsilon,
        )
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(
            config.embedding_dimension,
            config.rms_norm_epsilon,
        )
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))

    def forward_with_cache(
        self,
        inputs: Tensor,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attended, present_key_value = self.attention.forward_with_cache(
            self.attention_norm(inputs),
            past_key_value,
        )
        hidden = inputs + attended
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        return hidden, present_key_value


class TokiPonaTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocabulary_size,
            config.embedding_dimension,
        )
        self.position_embedding = nn.Embedding(
            config.context_size,
            config.embedding_dimension,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.layer_count)
        )
        self.final_norm = RMSNorm(
            config.embedding_dimension,
            config.rms_norm_epsilon,
        )
        self.unembedding = nn.Linear(
            config.embedding_dimension,
            config.vocabulary_size,
            bias=False,
        )

        self.apply(self._initialize_module)
        self.unembedding.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.context_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context size "
                f"{self.config.context_size}"
            )
        if targets is not None and targets.shape != token_ids.shape:
            raise ValueError("targets must have the same shape as token_ids")

        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.unembedding(self.final_norm(hidden))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, -1),
                targets.reshape(batch_size * sequence_length),
            )
        return logits, loss

    def forward_with_cache(
        self,
        token_ids: Tensor,
        past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None,
    ) -> tuple[Tensor, tuple[tuple[Tensor, Tensor], ...]]:
        """Run cached inference and return logits plus each layer's KV cache."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        batch_size, sequence_length = token_ids.shape
        if sequence_length == 0:
            raise ValueError("token_ids sequence must not be empty")

        if past_key_values is None:
            layer_pasts: tuple[tuple[Tensor, Tensor] | None, ...] = (
                None,
            ) * len(self.blocks)
            past_length = 0
        else:
            if len(past_key_values) != len(self.blocks):
                raise ValueError("past_key_values must have one entry per layer")
            layer_pasts = past_key_values
            past_lengths = {key.shape[-2] for key, _ in past_key_values}
            if len(past_lengths) != 1:
                raise ValueError("all cached layers must have the same sequence length")
            past_length = past_lengths.pop()

        total_length = past_length + sequence_length
        if total_length > self.config.context_size:
            raise ValueError(
                f"cached sequence length {total_length} exceeds context size "
                f"{self.config.context_size}"
            )

        positions = torch.arange(
            past_length,
            total_length,
            device=token_ids.device,
        )
        hidden = self.token_embedding(token_ids)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        present_key_values: list[tuple[Tensor, Tensor]] = []
        for block, layer_past in zip(self.blocks, layer_pasts, strict=True):
            hidden, present = block.forward_with_cache(hidden, layer_past)
            present_key_values.append(present)
        logits = self.unembedding(self.final_norm(hidden))
        return logits, tuple(present_key_values)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

