#!/usr/bin/env python3
"""Rebalance an organized kulupu-lapo split by tokenizer token counts.

The number of samples in each split remains fixed. Whole Markdown/text pairs
are reassigned so that the aggregate planned-tokenizer counts are as close as
possible to the requested 80/10/10 ratios.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from .analyze_tokenization import analyze_text
    from .split_kulupu_lapo import (
        DEFAULT_SEED,
        MANIFEST_NAME,
        SPLIT_RATIOS,
        remove_empty_source_directories,
        verify,
    )
except ImportError:  # Direct execution: python scripts/rebalance_kulupu_lapo.py
    from analyze_tokenization import analyze_text
    from split_kulupu_lapo import (
        DEFAULT_SEED,
        MANIFEST_NAME,
        SPLIT_RATIOS,
        remove_empty_source_directories,
        verify,
    )


@dataclass(frozen=True)
class Sample:
    md_path: Path
    txt_path: Path
    original_md: str
    tokens: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=script_dir.parent / "kulupu-lapo",
        help="organized dataset directory (default: kulupu-lapo in the project root)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed for deterministic tie-breaking (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report existing per-split token totals without changing files",
    )
    return parser.parse_args()


def read_manifest(dataset: Path) -> dict[str, object]:
    return json.loads((dataset / MANIFEST_NAME).read_text(encoding="utf-8"))


def read_samples(dataset: Path, manifest: dict[str, object]) -> list[Sample]:
    entries = manifest.get("samples")
    if not isinstance(entries, list):
        raise ValueError("manifest does not contain a samples list")

    samples: list[Sample] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest contains a malformed sample entry")
        split = entry.get("split")
        sample_id = entry.get("id")
        original_md = entry.get("original_md")
        if (
            split not in SPLIT_RATIOS
            or not isinstance(sample_id, str)
            or not isinstance(original_md, str)
        ):
            raise ValueError(f"invalid manifest entry: {entry!r}")

        sample_dir = dataset / split / sample_id
        md_path = sample_dir / f"{sample_id}.md"
        txt_path = sample_dir / f"{sample_id}.txt"
        text = txt_path.read_text(encoding="utf-8")
        samples.append(
            Sample(
                md_path=md_path,
                txt_path=txt_path,
                original_md=original_md,
                tokens=analyze_text(text).total,
            )
        )
    return samples


def existing_totals(samples: list[Sample], manifest: dict[str, object]) -> dict[str, int]:
    totals = {split: 0 for split in SPLIT_RATIOS}
    entries = manifest["samples"]
    assert isinstance(entries, list)
    for sample, entry in zip(samples, entries, strict=True):
        assert isinstance(entry, dict)
        split = entry["split"]
        assert isinstance(split, str)
        totals[split] += sample.tokens
    return totals


def objective(totals: dict[str, int], targets: dict[str, float]) -> float:
    return sum(abs(totals[split] - targets[split]) for split in SPLIT_RATIOS)


def allocate(
    samples: list[Sample],
    capacities: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[Sample]], dict[str, int], int]:
    """Allocate largest samples first, then improve the result with pair swaps."""
    rng = random.Random(seed)
    ordered = samples.copy()
    rng.shuffle(ordered)
    ordered.sort(key=lambda sample: sample.tokens, reverse=True)

    total_tokens = sum(sample.tokens for sample in samples)
    targets = {
        split: total_tokens * ratio for split, ratio in SPLIT_RATIOS.items()
    }
    groups: dict[str, list[Sample]] = {split: [] for split in SPLIT_RATIOS}
    totals = {split: 0 for split in SPLIT_RATIOS}

    for sample in ordered:
        eligible = [
            split
            for split in SPLIT_RATIOS
            if len(groups[split]) < capacities[split]
        ]
        tie_breakers = {split: rng.random() for split in eligible}
        chosen = min(
            eligible,
            key=lambda split: (
                (totals[split] + sample.tokens) / targets[split],
                tie_breakers[split],
            ),
        )
        groups[chosen].append(sample)
        totals[chosen] += sample.tokens

    swaps = 0
    split_names = list(SPLIT_RATIOS)
    for _ in range(100):
        before = objective(totals, targets)
        best: tuple[float, str, str, int, int] | None = None
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1 :]:
                unchanged = sum(
                    abs(totals[split] - targets[split])
                    for split in split_names
                    if split not in (left, right)
                )
                for left_sample_index, left_sample in enumerate(groups[left]):
                    for right_sample_index, right_sample in enumerate(groups[right]):
                        new_left = totals[left] - left_sample.tokens + right_sample.tokens
                        new_right = totals[right] - right_sample.tokens + left_sample.tokens
                        after = (
                            unchanged
                            + abs(new_left - targets[left])
                            + abs(new_right - targets[right])
                        )
                        if after >= before - 1e-9:
                            continue
                        if best is None or after < best[0]:
                            best = (
                                after,
                                left,
                                right,
                                left_sample_index,
                                right_sample_index,
                            )
        if best is None:
            break

        _, left, right, left_index, right_index = best
        left_sample = groups[left][left_index]
        right_sample = groups[right][right_index]
        groups[left][left_index], groups[right][right_index] = (
            right_sample,
            left_sample,
        )
        totals[left] += right_sample.tokens - left_sample.tokens
        totals[right] += left_sample.tokens - right_sample.tokens
        swaps += 1

    # The numbered IDs must not reveal the size-based allocation order.
    for split in SPLIT_RATIOS:
        rng.shuffle(groups[split])

    return groups, totals, swaps


def print_totals(totals: dict[str, int]) -> None:
    grand_total = sum(totals.values())
    for split in SPLIT_RATIOS:
        percentage = 100.0 * totals[split] / grand_total
        print(f"{split}: {totals[split]:,} tokens ({percentage:.6f}%)")
    print(f"total: {grand_total:,} tokens")


def rebalance(dataset: Path, seed: int) -> int:
    manifest_path = dataset / MANIFEST_NAME
    errors = verify(dataset, manifest_path)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    try:
        manifest = read_manifest(dataset)
        samples = read_samples(dataset, manifest)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: cannot read organized samples: {error}", file=sys.stderr)
        return 1

    counts = manifest.get("counts")
    if not isinstance(counts, dict) or set(counts) != set(SPLIT_RATIOS):
        print("error: manifest has invalid split counts", file=sys.stderr)
        return 1
    capacities = {split: int(counts[split]) for split in SPLIT_RATIOS}

    before = existing_totals(samples, manifest)
    print("before:")
    print_totals(before)

    groups, token_totals, swaps = allocate(samples, capacities, seed)
    if before == token_totals and manifest.get("assignment_method") == "token-balanced":
        print("dataset is already token-balanced")
        return 0

    staging_root = dataset / ".split-rebalance-staging"
    if staging_root.exists():
        print(f"error: staging path already exists: {staging_root}", file=sys.stderr)
        return 1

    staged: list[Sample] = []
    staging_root.mkdir()
    width = max(6, len(str(len(samples))))
    for number, sample in enumerate(samples, start=1):
        stage_id = f"{number:0{width}d}"
        stage_dir = staging_root / stage_id
        stage_dir.mkdir()
        staged_md = stage_dir / f"{stage_id}.md"
        staged_txt = stage_dir / f"{stage_id}.txt"
        sample.md_path.replace(staged_md)
        sample.txt_path.replace(staged_txt)
        staged.append(
            Sample(staged_md, staged_txt, sample.original_md, sample.tokens)
        )

    # Recreate the allocation with staged paths. The sample identity is its
    # original Markdown path, which is unique in this corpus.
    staged_by_original = {sample.original_md: sample for sample in staged}
    staged_groups = {
        split: [staged_by_original[sample.original_md] for sample in group]
        for split, group in groups.items()
    }
    remove_empty_source_directories(dataset)

    entries: list[dict[str, object]] = []
    id_width = max(4, len(str(max(capacities.values()))))
    for split in SPLIT_RATIOS:
        for number, sample in enumerate(staged_groups[split], start=1):
            sample_id = f"{number:0{id_width}d}"
            sample_dir = dataset / split / sample_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            sample.md_path.replace(sample_dir / f"{sample_id}.md")
            sample.txt_path.replace(sample_dir / f"{sample_id}.txt")
            entries.append(
                {
                    "split": split,
                    "id": sample_id,
                    "original_md": sample.original_md,
                    "tokens": sample.tokens,
                }
            )

    remove_empty_source_directories(dataset)
    grand_total = sum(token_totals.values())
    new_manifest = dict(manifest)
    new_manifest.update(
        {
            "version": 2,
            "seed": seed,
            "assignment_method": "token-balanced",
            "ratios": SPLIT_RATIOS,
            "counts": capacities,
            "token_counts": token_totals,
            "token_percentages": {
                split: 100.0 * token_totals[split] / grand_total
                for split in SPLIT_RATIOS
            },
            "samples": entries,
        }
    )
    temporary_manifest = dataset / f".{MANIFEST_NAME}.tmp"
    temporary_manifest.write_text(
        json.dumps(new_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)

    errors = verify(dataset, manifest_path)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"rebalanced using {swaps} local-improvement swaps")
    print("after:")
    print_totals(token_totals)
    return 0


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        print(f"error: dataset directory does not exist: {dataset}", file=sys.stderr)
        return 2

    if args.check:
        errors = verify(dataset, dataset / MANIFEST_NAME)
        if errors:
            for error in errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
        try:
            manifest = read_manifest(dataset)
            samples = read_samples(dataset, manifest)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            print(f"error: cannot count organized samples: {error}", file=sys.stderr)
            return 1
        print_totals(existing_totals(samples, manifest))
        return 0

    return rebalance(dataset, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
