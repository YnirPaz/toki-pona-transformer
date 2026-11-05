#!/usr/bin/env python3
"""Randomly organize cleaned kulupu-lapo samples into numbered data splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


SPLIT_RATIOS = {
    "training": 0.80,
    "validation": 0.10,
    "evaluation": 0.10,
}
MANIFEST_NAME = "split_manifest.json"
DEFAULT_SEED = 20261014


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=script_dir.parent / "kulupu-lapo",
        help="dataset directory (default: kulupu-lapo in the project root)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random seed used for assignment (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing split without moving or deleting files",
    )
    return parser.parse_args()


def split_counts(sample_count: int) -> dict[str, int]:
    validation = round(sample_count * SPLIT_RATIOS["validation"])
    evaluation = round(sample_count * SPLIT_RATIOS["evaluation"])
    return {
        "training": sample_count - validation - evaluation,
        "validation": validation,
        "evaluation": evaluation,
    }


def verify(dataset: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read manifest: {error}"]

    samples = manifest.get("samples")
    if not isinstance(samples, list):
        return ["manifest does not contain a samples list"]

    expected_files: set[Path] = set()
    observed_counts = {split: 0 for split in SPLIT_RATIOS}
    for entry in samples:
        if not isinstance(entry, dict):
            errors.append("manifest contains a malformed sample entry")
            continue
        split = entry.get("split")
        sample_id = entry.get("id")
        if split not in SPLIT_RATIOS or not isinstance(sample_id, str):
            errors.append(f"invalid manifest entry: {entry!r}")
            continue

        observed_counts[split] += 1
        sample_dir = dataset / split / sample_id
        md_path = sample_dir / f"{sample_id}.md"
        txt_path = sample_dir / f"{sample_id}.txt"
        expected_files.update((md_path.resolve(), txt_path.resolve()))

        if not md_path.is_file():
            errors.append(f"missing {md_path}")
        if not txt_path.is_file():
            errors.append(f"missing {txt_path}")
        if sample_dir.is_dir():
            unexpected = [
                child.name
                for child in sample_dir.iterdir()
                if child.resolve() not in {md_path.resolve(), txt_path.resolve()}
            ]
            if unexpected:
                errors.append(f"unexpected files in {sample_dir}: {', '.join(unexpected)}")

    declared_counts = manifest.get("counts")
    if declared_counts != observed_counts:
        errors.append(
            f"manifest counts {declared_counts!r} do not match entries {observed_counts!r}"
        )

    actual_files = {
        path.resolve()
        for split in SPLIT_RATIOS
        for path in (dataset / split).rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        missing = expected_files - actual_files
        extra = actual_files - expected_files
        if missing:
            errors.append(f"{len(missing)} expected split files are missing")
        if extra:
            errors.append(f"{len(extra)} unexpected files are present in split folders")

    outside_split_files = [
        path
        for path in dataset.rglob("*")
        if path.is_file()
        and path.name != MANIFEST_NAME
        and path.relative_to(dataset).parts[0] not in SPLIT_RATIOS
    ]
    if outside_split_files:
        errors.append(f"{len(outside_split_files)} data files remain outside split folders")

    return errors


def remove_empty_source_directories(dataset: Path) -> None:
    split_roots = {(dataset / split).resolve() for split in SPLIT_RATIOS}
    directories = sorted(
        (path for path in dataset.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        if directory.resolve() in split_roots:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass


def organize(dataset: Path, seed: int) -> int:
    manifest_path = dataset / MANIFEST_NAME
    split_paths = {split: dataset / split for split in SPLIT_RATIOS}
    existing_split_paths = [path for path in split_paths.values() if path.exists()]
    if manifest_path.exists() or existing_split_paths:
        errors = verify(dataset, manifest_path)
        if errors:
            for error in errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
        print("dataset is already split and passes verification")
        return 0

    txt_files = sorted(dataset.rglob("*.txt"), key=lambda path: path.as_posix().casefold())
    if not txt_files:
        print(f"error: no cleaned text samples found under {dataset}", file=sys.stderr)
        return 2

    samples: list[tuple[Path, Path]] = []
    for txt_path in txt_files:
        md_path = txt_path.with_suffix(".md")
        if not md_path.is_file():
            print(f"error: missing Markdown pair for {txt_path}", file=sys.stderr)
            return 1
        samples.append((md_path, txt_path))

    paired_markdown = {md.resolve() for md, _ in samples}
    rejected_sources = sorted(
        (
            md
            for md in dataset.rglob("*.md")
            if md.resolve() not in paired_markdown
        ),
        key=lambda path: path.as_posix().casefold(),
    )

    rng = random.Random(seed)
    rng.shuffle(samples)
    counts = split_counts(len(samples))
    width = max(4, len(str(max(counts.values()))))

    entries: list[dict[str, str]] = []
    offset = 0
    plan: list[tuple[Path, Path, Path, Path]] = []
    for split, count in counts.items():
        for number, (source_md, source_txt) in enumerate(
            samples[offset : offset + count], start=1
        ):
            sample_id = f"{number:0{width}d}"
            sample_dir = split_paths[split] / sample_id
            target_md = sample_dir / f"{sample_id}.md"
            target_txt = sample_dir / f"{sample_id}.txt"
            plan.append((source_md, source_txt, target_md, target_txt))
            entries.append(
                {
                    "split": split,
                    "id": sample_id,
                    "original_md": source_md.relative_to(dataset).as_posix(),
                }
            )
        offset += count

    targets = [target for _, _, md, txt in plan for target in (md, txt)]
    collisions = [target for target in targets if target.exists()]
    if collisions:
        print(f"error: {len(collisions)} planned targets already exist", file=sys.stderr)
        return 1

    for source_md, source_txt, target_md, target_txt in plan:
        target_md.parent.mkdir(parents=True, exist_ok=False)
        source_md.replace(target_md)
        source_txt.replace(target_txt)

    # These are sources rejected by the text-cleaning quality gates. They have
    # no usable .txt pair and therefore cannot be samples in any split.
    rejected_relative = [path.relative_to(dataset).as_posix() for path in rejected_sources]
    for source in rejected_sources:
        source.unlink()

    manifest = {
        "version": 1,
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "counts": counts,
        "rejected_sources_removed": rejected_relative,
        "samples": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    remove_empty_source_directories(dataset)

    errors = verify(dataset, manifest_path)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"organized {len(samples)} samples with seed {seed}: "
        + ", ".join(f"{split}={count}" for split, count in counts.items())
    )
    print(f"removed {len(rejected_sources)} rejected Markdown sources")
    return 0


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        print(f"error: dataset directory does not exist: {dataset}", file=sys.stderr)
        return 2

    manifest_path = dataset / MANIFEST_NAME
    if args.check:
        errors = verify(dataset, manifest_path)
        if errors:
            for error in errors:
                print(f"error: {error}", file=sys.stderr)
            return 1
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            "verified split: "
            + ", ".join(
                f"{split}={manifest['counts'][split]}" for split in SPLIT_RATIOS
            )
        )
        return 0

    return organize(dataset, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
