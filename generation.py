"""Autoregressive loading and generation for trained Toki Pona models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor

from model import ModelConfig, TokiPonaTransformer
from tokenizer import TokiPonaTokenizer
from training import CHECKPOINT_FORMAT, CHECKPOINT_VERSION


@dataclass(frozen=True)
class TokenProbability:
    token_id: int
    token: str
    probability: float


@dataclass(frozen=True)
class GenerationStep:
    token_id: int
    token: str
    text_piece: str
    probability: float
    top_probabilities: tuple[TokenProbability, ...]


def _automatic_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint(run_directory: Path, checkpoint: str | Path) -> Path:
    if str(checkpoint) == "latest":
        path = run_directory / "checkpoint.pt"
    elif str(checkpoint) == "best":
        candidates: list[tuple[int, Path]] = []
        for path in (run_directory / "best").glob("epoch_*.pt"):
            try:
                epoch = int(path.stem.removeprefix("epoch_"))
            except ValueError:
                continue
            candidates.append((epoch, path))
        if not candidates:
            raise FileNotFoundError(f"no best checkpoints found in {run_directory / 'best'}")
        return max(candidates, key=lambda item: item[0])[1]
    else:
        supplied = Path(checkpoint)
        path = supplied if supplied.is_absolute() else run_directory / supplied
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path


class AutoregressiveGenerator:
    """A loaded model with temperature sampling and cached decoding."""

    def __init__(
        self,
        model: TokiPonaTransformer,
        tokenizer: TokiPonaTokenizer,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        run_directory: str | Path,
        *,
        checkpoint: str | Path = "best",
        device: str | torch.device | None = None,
    ) -> "AutoregressiveGenerator":
        run_path = Path(run_directory)
        tokenizer = TokiPonaTokenizer.load(run_path / "tokenizer.json")
        checkpoint_path = _resolve_checkpoint(run_path, checkpoint)
        payload: Any = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must contain a dictionary")
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unrecognized training checkpoint format")
        if payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version: {payload.get('version')!r}")
        if payload.get("tokenizer_vocabulary") != list(tokenizer.vocabulary):
            raise ValueError("checkpoint tokenizer vocabulary does not match")

        config_payload = payload.get("model_config")
        if not isinstance(config_payload, dict):
            raise ValueError("checkpoint has no valid model configuration")
        config_payload = dict(config_payload)
        config_payload.setdefault("dropout_probability", 0.0)
        config = ModelConfig(**config_payload)
        if config.vocabulary_size != len(tokenizer):
            raise ValueError("checkpoint model vocabulary size does not match")

        state_dict = payload.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint has no model state")
        model = TokiPonaTransformer(config)
        model.load_state_dict(state_dict)
        return cls(model, tokenizer, device=device or _automatic_device())

    def encode_prompt(self, text: str) -> list[int]:
        """Encode a prompt with BOS but without the training-only trailing EOS."""
        return self.tokenizer.encode(text, add_bos=True, add_eos=False)

    def token_label(self, token_id: int) -> str:
        """Return a compact, human-readable label for one vocabulary item."""
        if token_id in self.tokenizer.id_to_word:
            return self.tokenizer.id_to_word[token_id]
        if token_id in self.tokenizer.id_to_letter:
            return self.tokenizer.id_to_letter[token_id]
        if token_id in self.tokenizer.id_to_mark:
            mark = self.tokenizer.id_to_mark[token_id]
            return "\\n" if mark == "\n" else mark
        return self.tokenizer.vocabulary[token_id]

    def _sample(
        self,
        logits: Tensor,
        temperature: float,
        top_count: int,
    ) -> tuple[int, float, tuple[TokenProbability, ...]]:
        probabilities = torch.softmax(logits.float() / temperature, dim=-1)
        selected_id = int(torch.multinomial(probabilities, num_samples=1).item())
        selected_probability = float(probabilities[selected_id].item())
        count = min(top_count, probabilities.numel())
        top_values, top_ids = torch.topk(probabilities, count)
        top = tuple(
            TokenProbability(
                token_id=int(token_id),
                token=self.token_label(int(token_id)),
                probability=float(probability),
            )
            for probability, token_id in zip(
                top_values.tolist(),
                top_ids.tolist(),
                strict=True,
            )
        )
        return selected_id, selected_probability, top

    @torch.inference_mode()
    def generate(
        self,
        token_ids: list[int],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 256,
        top_count: int = 5,
    ) -> Iterator[GenerationStep]:
        """Yield sampled tokens until EOS or ``max_new_tokens`` is reached."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")
        if top_count <= 0:
            raise ValueError("top_count must be positive")
        if not token_ids:
            token_ids = [self.tokenizer.bos_id]
        if any(
            not isinstance(token_id, int) or not 0 <= token_id < len(self.tokenizer)
            for token_id in token_ids
        ):
            raise ValueError("token_ids contains an invalid token ID")

        all_token_ids = list(token_ids)
        context_size = self.model.config.context_size
        model_input = all_token_ids[-context_size:]
        input_tensor = torch.tensor(
            [model_input],
            dtype=torch.long,
            device=self.device,
        )
        logits, cache = self.model.forward_with_cache(input_tensor)

        for _ in range(max_new_tokens):
            selected_id, selected_probability, top = self._sample(
                logits[0, -1],
                temperature,
                top_count,
            )
            before_text = self.tokenizer.decode(all_token_ids)
            all_token_ids.append(selected_id)
            after_text = self.tokenizer.decode(all_token_ids)
            text_piece = (
                after_text[len(before_text) :]
                if after_text.startswith(before_text)
                else self.tokenizer.decode([selected_id])
            )
            yield GenerationStep(
                token_id=selected_id,
                token=self.token_label(selected_id),
                text_piece=text_piece,
                probability=selected_probability,
                top_probabilities=top,
            )
            if selected_id == self.tokenizer.eos_id:
                return

            cached_length = cache[0][0].shape[-2]
            if cached_length < context_size:
                next_input = torch.tensor(
                    [[selected_id]],
                    dtype=torch.long,
                    device=self.device,
                )
                logits, cache = self.model.forward_with_cache(next_input, cache)
            else:
                window = all_token_ids[-context_size:]
                input_tensor = torch.tensor(
                    [window],
                    dtype=torch.long,
                    device=self.device,
                )
                logits, cache = self.model.forward_with_cache(input_tensor)
