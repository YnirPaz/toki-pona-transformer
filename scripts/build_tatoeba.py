#!/usr/bin/env python3
"""Build a deduplicated, kulupu-lapo-shaped dataset from Tatoeba Parquet files.

Only the ``tok`` text and its Tatoeba sentence ID are retained. Samples are
deduplicated by their case-insensitive Unicode word sequence, and any sample
whose word sequence occurs in the reference corpus is omitted. The latter
also catches a Tatoeba sentence embedded in a longer kulupu-lapo document.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:
    from .analyze_tokenization import analyze_text
    from .split_kulupu_lapo import SPLIT_RATIOS, verify
except ImportError:  # Direct execution: python scripts/build_tatoeba.py
    from analyze_tokenization import analyze_text
    from split_kulupu_lapo import SPLIT_RATIOS, verify


DEFAULT_SEED = 20261024
MANIFEST_NAME = "split_manifest.json"
WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
END = object()


@dataclass(frozen=True)
class Sample:
    tatoeba_id: int
    text: str
    key: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=project / "tatoeba-tokipona",
        help="downloaded Tatoeba dataset directory",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=project / "tatoeba-tokipona-processed",
        help="destination dataset directory",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=project / "kulupu-lapo",
        help="corpus against which Tatoeba samples are deduplicated",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing output without changing it",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    """Normalize storage whitespace without changing words or punctuation."""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def word_key(text: str) -> tuple[str, ...]:
    """Return the comparison form used for all duplicate decisions."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    return tuple(WORD_RE.findall(normalized))


def read_tatoeba(source: Path) -> tuple[list[Sample], dict[str, int]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit(
            "error: PyArrow is required to read the Tatoeba Parquet files "
            "(install it with: python -m pip install pyarrow)"
        ) from error

    parquet_files = sorted((source / "data").glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"no Parquet files found under {source / 'data'}")

    by_key: dict[tuple[str, ...], Sample] = {}
    input_rows = empty_rows = 0
    for parquet_file in parquet_files:
        parquet_source = parquet.ParquetFile(parquet_file)
        for batch in parquet_source.iter_batches(columns=["target_id", "tok"]):
            data = batch.to_pydict()
            for tatoeba_id, raw_text in zip(
                data["target_id"], data["tok"], strict=True
            ):
                input_rows += 1
                text = clean_text(raw_text or "")
                key = word_key(text)
                if not key:
                    empty_rows += 1
                    continue
                sample = Sample(int(tatoeba_id), text, key)
                previous = by_key.get(key)
                if previous is None or (sample.tatoeba_id, sample.text) < (
                    previous.tatoeba_id,
                    previous.text,
                ):
                    by_key[key] = sample

    samples = sorted(by_key.values(), key=lambda item: (item.tatoeba_id, item.text))
    stats = {
        "input_rows": input_rows,
        "empty_rows_removed": empty_rows,
        "within_tatoeba_duplicates_removed": input_rows - empty_rows - len(samples),
    }
    return samples, stats


def reference_matches(samples: list[Sample], reference: Path) -> set[tuple[str, ...]]:
    """Find candidate word sequences occurring anywhere in reference documents."""
    reference_files = sorted(reference.rglob("*.txt"))
    if not reference_files:
        raise ValueError(f"no reference .txt files found under {reference}")

    trie: dict[object, dict] = {}
    for sample in samples:
        node = trie
        for word in sample.key:
            node = node.setdefault(word, {})
        node[END] = sample.key

    matches: set[tuple[str, ...]] = set()
    for path in reference_files:
        words = word_key(path.read_text(encoding="utf-8"))
        for start in range(len(words)):
            node = trie
            for word in words[start:]:
                child = node.get(word)
                if child is None:
                    break
                node = child
                match = node.get(END)
                if match is not None:
                    matches.add(match)
    return matches


def split_counts(sample_count: int) -> dict[str, int]:
    validation = round(sample_count * SPLIT_RATIOS["validation"])
    evaluation = round(sample_count * SPLIT_RATIOS["evaluation"])
    return {
        "training": sample_count - validation - evaluation,
        "validation": validation,
        "evaluation": evaluation,
    }


def markdown_source(sample: Sample) -> str:
    return (
        "---\n"
        f"title: Tatoeba sentence {sample.tatoeba_id}\n"
        "description: null\n"
        "authors: null\n"
        "proofreaders: null\n"
        "date: null\n"
        "date-precision: null\n"
        "tags:\n"
        "  - Tatoeba\n"
        "license: CC BY 2.0\n"
        "sources:\n"
        f"  - https://tatoeba.org/en/sentences/show/{sample.tatoeba_id}\n"
        "archives: null\n"
        "preprocessing: extracted from the tok field; whitespace normalized\n"
        "accessibility-notes: null\n"
        "notes: translations omitted\n"
        "---\n"
        f"{sample.text}\n"
    )


def build(
    source: Path,
    output: Path,
    reference: Path,
    seed: int,
) -> int:
    if output.exists():
        print(f"error: output already exists: {output}", file=sys.stderr)
        return 1

    samples, stats = read_tatoeba(source)
    duplicates = reference_matches(samples, reference)
    samples = [sample for sample in samples if sample.key not in duplicates]
    stats["cross_dataset_duplicates_removed"] = len(duplicates)
    stats["samples_written"] = len(samples)

    random.Random(seed).shuffle(samples)
    counts = split_counts(len(samples))
    width = max(4, len(str(max(counts.values()))))
    entries: list[dict[str, object]] = []
    offset = 0

    output.mkdir(parents=True)
    try:
        for split, count in counts.items():
            split_samples = samples[offset : offset + count]
            for number, sample in enumerate(split_samples, start=1):
                sample_id = f"{number:0{width}d}"
                sample_dir = output / split / sample_id
                sample_dir.mkdir(parents=True)
                (sample_dir / f"{sample_id}.md").write_text(
                    markdown_source(sample), encoding="utf-8", newline="\n"
                )
                (sample_dir / f"{sample_id}.txt").write_text(
                    sample.text + "\n", encoding="utf-8", newline="\n"
                )
                entries.append(
                    {
                        "split": split,
                        "id": sample_id,
                        "original_md": f"tatoeba/{sample.tatoeba_id}.md",
                        "tatoeba_id": sample.tatoeba_id,
                        "tokens": analyze_text(sample.text).total,
                    }
                )
                if len(entries) % 5000 == 0:
                    print(f"wrote {len(entries):,}/{len(samples):,} samples", flush=True)
            offset += count

        manifest = {
            "version": 2,
            "source": "NetherQuartz/tatoeba-tokipona",
            "source_license": "CC BY 2.0",
            "seed": seed,
            "ratios": SPLIT_RATIOS,
            "counts": counts,
            "assignment_method": "deterministic-random",
            "deduplication": {
                "comparison": "NFC casefolded Unicode word sequence",
                "reference": reference.name,
                "reference_scope": "contiguous sequence anywhere in each text sample",
            },
            "statistics": stats,
            "samples": entries,
        }
        (output / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except BaseException:
        print(
            f"error: incomplete output remains at {output}; remove it before retrying",
            file=sys.stderr,
        )
        raise

    errors = verify(output, output / MANIFEST_NAME)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"built {len(samples):,} samples: "
        + ", ".join(f"{name}={count:,}" for name, count in counts.items())
    )
    print(
        "removed "
        f"{stats['within_tatoeba_duplicates_removed']:,} within-Tatoeba duplicates and "
        f"{stats['cross_dataset_duplicates_removed']:,} kulupu-lapo overlaps"
    )
    return 0


def check(output: Path, reference: Path) -> int:
    errors = verify(output, output / MANIFEST_NAME)
    text_files = sorted(
        path for split in SPLIT_RATIOS for path in (output / split).rglob("*.txt")
    )
    keys: set[tuple[str, ...]] = set()
    samples: list[Sample] = []
    for number, path in enumerate(text_files):
        text = path.read_text(encoding="utf-8")
        key = word_key(text)
        if not key:
            errors.append(f"empty normalized sample: {path}")
        elif key in keys:
            errors.append(f"duplicate normalized sample: {path}")
        else:
            keys.add(key)
            samples.append(Sample(number, text, key))

    overlaps = reference_matches(samples, reference) if samples else set()
    if overlaps:
        errors.append(f"{len(overlaps)} samples overlap the reference corpus")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"verified {len(samples):,} unique samples with no {reference.name} overlap"
    )
    return 0


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    reference = args.reference.resolve()
    if args.check:
        return check(output, reference)
    return build(source, output, reference, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
