"""Build token streams from the training, validation, and evaluation splits."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path

from tokenizer import TokiPonaTokenizer


VALIDATION_STREAM_SEED = 20261015
EVALUATION_STREAM_SEED = 20261016
CONTEXT_SIZE = 256
DatasetRoot = str | Path
DatasetRoots = DatasetRoot | Sequence[DatasetRoot]


def _dataset_roots(datasets: DatasetRoots) -> tuple[Path, ...]:
    if isinstance(datasets, (str, Path)):
        return (Path(datasets),)
    roots = tuple(Path(dataset) for dataset in datasets)
    if not roots:
        raise ValueError("at least one dataset root is required")
    return roots


def _split_files(datasets: DatasetRoots, split: str) -> list[Path]:
    files: list[Path] = []
    for dataset in _dataset_roots(datasets):
        split_directory = dataset / split
        if not split_directory.is_dir():
            raise FileNotFoundError(f"dataset split does not exist: {split_directory}")
        files.extend(split_directory.rglob("*.txt"))
    files.sort(key=lambda path: path.as_posix().casefold())
    if not files:
        raise ValueError(f"combined dataset split contains no .txt files: {split}")
    return files


def _encode_files(files: Sequence[Path], tokenizer: TokiPonaTokenizer) -> list[int]:
    stream: list[int] = []
    for source in files:
        stream.extend(tokenizer.encode(source.read_text(encoding="utf-8")))
    return stream


def build_training_stream(
    datasets: DatasetRoots,
    tokenizer: TokiPonaTokenizer,
    *,
    rng: random.Random | None = None,
) -> list[int]:
    """Build one epoch in a newly shuffled document order.

    Pass the training loop's ``random.Random`` instance to make a complete run
    reproducible. If omitted, a newly system-seeded generator is used.
    """
    files = _split_files(datasets, "training")
    (rng if rng is not None else random.Random()).shuffle(files)
    return _encode_files(files, tokenizer)


def _build_fixed_stream(
    datasets: DatasetRoots,
    split: str,
    seed: int,
    tokenizer: TokiPonaTokenizer,
) -> list[int]:
    files = _split_files(datasets, split)
    random.Random(seed).shuffle(files)
    return _encode_files(files, tokenizer)


def build_validation_stream(
    datasets: DatasetRoots,
    tokenizer: TokiPonaTokenizer,
) -> list[int]:
    """Build the validation stream using its fixed hard-coded seed."""
    return _build_fixed_stream(datasets, "validation", VALIDATION_STREAM_SEED, tokenizer)


def build_evaluation_stream(
    datasets: DatasetRoots,
    tokenizer: TokiPonaTokenizer,
) -> list[int]:
    """Build the evaluation stream using its fixed hard-coded seed."""
    return _build_fixed_stream(datasets, "evaluation", EVALUATION_STREAM_SEED, tokenizer)


def iter_causal_windows(
    stream: Sequence[int],
    context_size: int = CONTEXT_SIZE,
) -> Iterator[tuple[Sequence[int], Sequence[int]]]:
    """Yield non-overlapping ``(input, target)`` windows and drop the remainder."""
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    for start in range(0, len(stream) - context_size, context_size):
        window = stream[start : start + context_size + 1]
        if len(window) != context_size + 1:
            break
        yield window[:-1], window[1:]

