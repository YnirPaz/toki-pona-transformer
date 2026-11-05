"""Training loop and checkpoint management for the transformer models."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import torch
from torch import Tensor

from model import ModelConfig, TokiPonaTransformer, model_config_for_size
from streams import CONTEXT_SIZE, build_training_stream, build_validation_stream
from tokenizer import TokiPonaTokenizer


CHECKPOINT_FORMAT = "toki-pona-transformer-training"
CHECKPOINT_VERSION = 1
TRAINING_STREAM_SEED = 20261018


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    maximum_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    checkpoint_interval_chunks: int = 1024
    random_seed: int = 20261018


def _causal_chunks(stream: list[int], context_size: int) -> tuple[Tensor, Tensor]:
    token_tensor = torch.tensor(stream, dtype=torch.long)
    chunk_count = (len(stream) - 1) // context_size
    if chunk_count == 0:
        raise ValueError("token stream is too short for one causal context")
    used_tokens = chunk_count * context_size
    inputs = token_tensor[:used_tokens].view(chunk_count, context_size)
    targets = token_tensor[1 : used_tokens + 1].view(chunk_count, context_size)
    return inputs, targets


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_payload(
    model: TokiPonaTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: TokiPonaTokenizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    epoch_index: int,
    chunks_completed: int,
    total_chunks: int,
    training_loss_sum: float,
    training_token_count: int,
    history: list[dict[str, Any]],
    best_validation_loss: float,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "tokenizer_vocabulary": list(tokenizer.vocabulary),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "progress": {
            "epoch_index": epoch_index,
            "chunks_completed": chunks_completed,
            "total_chunks": total_chunks,
            "training_loss_sum": training_loss_sum,
            "training_token_count": training_token_count,
        },
        "history": history,
        "best_validation_loss": best_validation_loss,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _save_checkpoint(
    path: Path,
    model: TokiPonaTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: TokiPonaTokenizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    **progress: Any,
) -> None:
    payload = _checkpoint_payload(
        model,
        optimizer,
        tokenizer,
        model_config,
        training_config,
        **progress,
    )
    _atomic_torch_save(payload, path)


def _load_checkpoint(
    path: Path,
    device: torch.device,
    model: TokiPonaTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: TokiPonaTokenizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unrecognized training checkpoint format")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {checkpoint.get('version')!r}")
    saved_model_config = checkpoint.get("model_config")
    if isinstance(saved_model_config, dict):
        # Checkpoints created before dropout support implicitly used zero dropout.
        saved_model_config = dict(saved_model_config)
        saved_model_config.setdefault("dropout_probability", 0.0)
    if saved_model_config != asdict(model_config):
        raise ValueError("checkpoint model configuration does not match")
    if checkpoint.get("tokenizer_vocabulary") != list(tokenizer.vocabulary):
        raise ValueError("checkpoint tokenizer vocabulary does not match")

    saved_training_config = checkpoint.get("training_config")
    if not isinstance(saved_training_config, dict):
        raise ValueError("checkpoint has no valid training configuration")
    current_training_config = asdict(training_config)
    for key, value in current_training_config.items():
        if key == "maximum_epochs":
            continue
        if saved_training_config.get(key) != value:
            raise ValueError(f"checkpoint training setting does not match: {key}")

    progress = checkpoint.get("progress")
    history = checkpoint.get("history")
    if not isinstance(progress, dict) or not isinstance(history, list):
        raise ValueError("checkpoint progress or history is malformed")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_state = checkpoint.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    return progress, history, float(checkpoint["best_validation_loss"])


@torch.no_grad()
def _validation_loss(
    model: TokiPonaTransformer,
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        batch_targets = targets[start : start + batch_size].to(device)
        _, loss = model(batch_inputs, batch_targets)
        assert loss is not None
        batch_tokens = batch_targets.numel()
        loss_sum += loss.item() * batch_tokens
        token_count += batch_tokens
    return loss_sum / token_count


def _history_payload(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    history: list[dict[str, Any]],
    best_validation_loss: float,
) -> dict[str, Any]:
    return {
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "best_validation_loss": best_validation_loss,
        "epochs": history,
    }


def train(
    datasets: str | Path | Sequence[str | Path],
    output_directory: str | Path,
    *,
    model_size: str = "small",
    resume_checkpoint: str | Path | None = None,
    maximum_epochs: int = 50,
    resume_epochs: int | None = None,
) -> None:
    if resume_epochs is not None:
        if resume_checkpoint is None:
            raise ValueError("resume_epochs requires a resume checkpoint")
        if resume_epochs <= 0:
            raise ValueError("resume_epochs must be positive")

    if isinstance(datasets, (str, Path)):
        dataset_roots = (Path(datasets).resolve(),)
    else:
        dataset_roots = tuple(Path(dataset).resolve() for dataset in datasets)
        if not dataset_roots:
            raise ValueError("at least one dataset root is required")
    output_directory = Path(output_directory).resolve()
    checkpoint_path = output_directory / "checkpoint.pt"
    history_path = output_directory / "history.json"
    tokenizer_path = output_directory / "tokenizer.json"

    training_config = TrainingConfig(maximum_epochs=maximum_epochs)
    tokenizer = TokiPonaTokenizer()
    model_config = model_config_for_size(
        model_size,
        len(tokenizer),
        context_size=CONTEXT_SIZE,
    )
    device = _select_device()

    if resume_checkpoint is None:
        occupied = [path for path in (checkpoint_path, history_path, tokenizer_path) if path.exists()]
        if occupied:
            names = ", ".join(path.name for path in occupied)
            raise FileExistsError(
                f"output directory already contains training state ({names}); "
                "resume it or choose another output directory"
            )
        output_directory.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tokenizer_path)
    else:
        resume_checkpoint = Path(resume_checkpoint).resolve()
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"missing tokenizer checkpoint: {tokenizer_path}")
        TokiPonaTokenizer.load(tokenizer_path)

    random.seed(training_config.random_seed)
    torch.manual_seed(training_config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.random_seed)

    model = TokiPonaTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    start_epoch = 0
    resume_chunks = 0
    resume_loss_sum = 0.0
    resume_token_count = 0
    resume_total_chunks = 0

    if resume_checkpoint is not None:
        progress, history, best_validation_loss = _load_checkpoint(
            resume_checkpoint,
            device,
            model,
            optimizer,
            tokenizer,
            model_config,
            training_config,
        )
        start_epoch = int(progress["epoch_index"])
        resume_chunks = int(progress["chunks_completed"])
        resume_total_chunks = int(progress["total_chunks"])
        resume_loss_sum = float(progress["training_loss_sum"])
        resume_token_count = int(progress["training_token_count"])
        if resume_epochs is not None:
            training_config = replace(
                training_config,
                maximum_epochs=start_epoch + resume_epochs,
            )
        print(
            f"Resuming {resume_checkpoint} at epoch {start_epoch + 1}, "
            f"chunk {resume_chunks:,}.",
            flush=True,
        )

    validation_stream = build_validation_stream(dataset_roots, tokenizer)
    validation_inputs, validation_targets = _causal_chunks(
        validation_stream,
        model_config.context_size,
    )
    del validation_stream

    print(
        f"Model: {model_size} | device: {device} | "
        f"parameters: {model.parameter_count():,} | "
        f"vocabulary: {len(tokenizer)} | batch: {training_config.batch_size} | "
        f"maximum epochs: {training_config.maximum_epochs}",
        flush=True,
    )

    if start_epoch >= training_config.maximum_epochs:
        print("The checkpoint has already reached the requested epoch limit.", flush=True)
        return

    training_started = time.monotonic()
    for epoch_index in range(start_epoch, training_config.maximum_epochs):
        epoch_started = time.monotonic()
        epoch_rng = random.Random(TRAINING_STREAM_SEED + epoch_index)
        training_stream = build_training_stream(dataset_roots, tokenizer, rng=epoch_rng)
        training_inputs, training_targets = _causal_chunks(
            training_stream,
            model_config.context_size,
        )
        del training_stream
        total_chunks = training_inputs.shape[0]

        chunks_completed = resume_chunks if epoch_index == start_epoch else 0
        loss_sum = resume_loss_sum if epoch_index == start_epoch else 0.0
        token_count = resume_token_count if epoch_index == start_epoch else 0
        if chunks_completed:
            if resume_total_chunks != total_chunks:
                raise ValueError(
                    "the reconstructed training stream has a different number of chunks"
                )
            if chunks_completed > total_chunks:
                raise ValueError("checkpoint progress exceeds the reconstructed epoch")

        next_checkpoint = (
            chunks_completed // training_config.checkpoint_interval_chunks + 1
        ) * training_config.checkpoint_interval_chunks
        model.train()

        for start in range(chunks_completed, total_chunks, training_config.batch_size):
            end = min(start + training_config.batch_size, total_chunks)
            batch_inputs = training_inputs[start:end].to(device)
            batch_targets = training_targets[start:end].to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(batch_inputs, batch_targets)
            assert loss is not None
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch_index + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                training_config.gradient_clip_norm,
            )
            optimizer.step()

            batch_tokens = batch_targets.numel()
            loss_sum += loss.item() * batch_tokens
            token_count += batch_tokens
            chunks_completed = end

            if chunks_completed >= next_checkpoint:
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    tokenizer,
                    model_config,
                    training_config,
                    epoch_index=epoch_index,
                    chunks_completed=chunks_completed,
                    total_chunks=total_chunks,
                    training_loss_sum=loss_sum,
                    training_token_count=token_count,
                    history=history,
                    best_validation_loss=best_validation_loss,
                )
                print(
                    f"Checkpoint | epoch {epoch_index + 1}/"
                    f"{training_config.maximum_epochs} | chunks "
                    f"{chunks_completed:,}/{total_chunks:,} | running loss "
                    f"{loss_sum / token_count:.6f}",
                    flush=True,
                )
                next_checkpoint += training_config.checkpoint_interval_chunks

        # This protects the completed training pass if validation is interrupted.
        _save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            tokenizer,
            model_config,
            training_config,
            epoch_index=epoch_index,
            chunks_completed=total_chunks,
            total_chunks=total_chunks,
            training_loss_sum=loss_sum,
            training_token_count=token_count,
            history=history,
            best_validation_loss=best_validation_loss,
        )
        print(
            f"Checkpoint | epoch {epoch_index + 1}/"
            f"{training_config.maximum_epochs} | training pass complete; "
            "starting validation",
            flush=True,
        )

        training_loss = loss_sum / token_count
        validation_loss = _validation_loss(
            model,
            validation_inputs,
            validation_targets,
            training_config.batch_size,
            device,
        )
        epoch_seconds = time.monotonic() - epoch_started
        record = {
            "epoch": epoch_index + 1,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "chunks": total_chunks,
            "seconds": epoch_seconds,
        }
        history.append(record)

        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = validation_loss

        next_epoch_progress = {
            "epoch_index": epoch_index + 1,
            "chunks_completed": 0,
            "total_chunks": 0,
            "training_loss_sum": 0.0,
            "training_token_count": 0,
            "history": history,
            "best_validation_loss": best_validation_loss,
        }
        _save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            tokenizer,
            model_config,
            training_config,
            **next_epoch_progress,
        )
        _atomic_json_save(
            _history_payload(
                model_config,
                training_config,
                history,
                best_validation_loss,
            ),
            history_path,
        )

        best_message = ""
        if is_best:
            best_path = output_directory / "best" / f"epoch_{epoch_index + 1:03d}.pt"
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                tokenizer,
                model_config,
                training_config,
                **next_epoch_progress,
            )
            best_message = f" | new best saved to {best_path.name}"

        elapsed = timedelta(seconds=round(time.monotonic() - training_started))
        print(
            f"Epoch {epoch_index + 1}/{training_config.maximum_epochs} | "
            f"train {training_loss:.6f} | validation {validation_loss:.6f} | "
            f"epoch {timedelta(seconds=round(epoch_seconds))} | total {elapsed}"
            f"{best_message}",
            flush=True,
        )

        resume_chunks = 0
        resume_loss_sum = 0.0
        resume_token_count = 0
        resume_total_chunks = 0

