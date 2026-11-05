"""Evaluate trained model runs on the fixed evaluation split."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
if str(PROJECT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIRECTORY))

from model import MODEL_SIZES, ModelConfig, TokiPonaTransformer  # noqa: E402
from streams import build_evaluation_stream  # noqa: E402
from tokenizer import TokiPonaTokenizer  # noqa: E402
from training import CHECKPOINT_FORMAT, CHECKPOINT_VERSION  # noqa: E402


DEFAULT_RUNS = tuple(PROJECT_DIRECTORY / "runs" / name for name in MODEL_SIZES)
BEST_CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d+)\.pt$")


@dataclass(frozen=True)
class EvaluationResult:
    run: str
    checkpoint: str
    epoch: int
    loss: float
    perplexity: float
    accuracy: float
    tokens: int


def select_checkpoint(run_directory: str | Path, kind: str = "best") -> Path:
    """Select the latest checkpoint or the highest-epoch best checkpoint."""
    run_path = Path(run_directory)
    if kind == "latest":
        checkpoint = run_path / "checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        return checkpoint
    if kind != "best":
        raise ValueError("checkpoint kind must be 'best' or 'latest'")

    candidates: list[tuple[int, Path]] = []
    for path in (run_path / "best").glob("epoch_*.pt"):
        match = BEST_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"no best checkpoints found in {run_path / 'best'}")
    return max(candidates, key=lambda item: item[0])[1]


def _load_model(
    run_directory: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TokiPonaTransformer, TokiPonaTokenizer, int]:
    tokenizer = TokiPonaTokenizer.load(run_directory / "tokenizer.json")
    checkpoint: Any = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a dictionary: {checkpoint_path}")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unrecognized checkpoint format: {checkpoint_path}")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {checkpoint_path}")
    if checkpoint.get("tokenizer_vocabulary") != list(tokenizer.vocabulary):
        raise ValueError(f"tokenizer vocabulary does not match: {checkpoint_path}")

    config_payload = checkpoint.get("model_config")
    if not isinstance(config_payload, dict):
        raise ValueError(f"checkpoint has no model configuration: {checkpoint_path}")
    # Older checkpoints predate the dropout field and implicitly used zero dropout.
    config_payload = dict(config_payload)
    config_payload.setdefault("dropout_probability", 0.0)
    config = ModelConfig(**config_payload)
    if config.vocabulary_size != len(tokenizer):
        raise ValueError(f"model vocabulary size does not match: {checkpoint_path}")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint has no model state: {checkpoint_path}")
    model = TokiPonaTransformer(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    progress = checkpoint.get("progress")
    epoch = int(progress.get("epoch_index", 0)) if isinstance(progress, dict) else 0
    return model, tokenizer, epoch


@torch.inference_mode()
def evaluate_run(
    run_directory: str | Path,
    dataset: str | Path,
    *,
    checkpoint_kind: str = "best",
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> EvaluationResult:
    """Evaluate one run and return token-weighted next-token metrics."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    run_path = Path(run_directory)
    checkpoint_path = select_checkpoint(run_path, checkpoint_kind)
    torch_device = torch.device(device)
    model, tokenizer, epoch = _load_model(run_path, checkpoint_path, torch_device)

    stream = build_evaluation_stream(dataset, tokenizer)
    context_size = model.config.context_size
    window_count = (len(stream) - 1) // context_size
    if window_count == 0:
        raise ValueError("evaluation stream is too short for one causal context")
    used_tokens = window_count * context_size
    token_tensor = torch.tensor(stream[: used_tokens + 1], dtype=torch.long)
    inputs = token_tensor[:used_tokens].view(window_count, context_size)
    targets = token_tensor[1:].view(window_count, context_size)

    loss_sum = 0.0
    correct = 0
    token_count = targets.numel()
    for start in range(0, window_count, batch_size):
        batch_inputs = inputs[start : start + batch_size].to(torch_device)
        batch_targets = targets[start : start + batch_size].to(torch_device)
        logits, _ = model(batch_inputs)
        loss_sum += F.cross_entropy(
            logits.flatten(0, 1),
            batch_targets.flatten(),
            reduction="sum",
        ).item()
        correct += (logits.argmax(dim=-1) == batch_targets).sum().item()

    loss = loss_sum / token_count
    return EvaluationResult(
        run=run_path.name,
        checkpoint=str(checkpoint_path),
        epoch=epoch,
        loss=loss,
        perplexity=math.exp(loss),
        accuracy=correct / token_count,
        tokens=token_count,
    )


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_DIRECTORY / "data" / "processed" / "kulupu-lapo",
        help="dataset root (default: data/processed/kulupu-lapo)",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=DEFAULT_RUNS,
        metavar="RUN_DIR",
        help="run directories to evaluate (default: all four runs)",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="checkpoint from each run to evaluate (default: best)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, such as auto, cpu, or cuda (default: auto)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="also save the results as JSON",
    )
    return parser.parse_args()


def _print_results(results: list[EvaluationResult]) -> None:
    headers = ("model", "epoch", "loss", "perplexity", "accuracy", "tokens")
    rows = [
        (
            result.run,
            str(result.epoch),
            f"{result.loss:.6f}",
            f"{result.perplexity:.3f}",
            f"{result.accuracy:.2%}",
            f"{result.tokens:,}",
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    args = parse_args()
    try:
        device = _select_device(args.device)
        results = []
        for run in args.runs:
            print(f"Evaluating {run.name} on {device}...", flush=True)
            results.append(
                evaluate_run(
                    run,
                    args.dataset,
                    checkpoint_kind=args.checkpoint,
                    batch_size=args.batch_size,
                    device=device,
                )
            )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print()
    _print_results(results)
    if args.json is not None:
        output_path = args.json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([asdict(result) for result in results], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved JSON results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
