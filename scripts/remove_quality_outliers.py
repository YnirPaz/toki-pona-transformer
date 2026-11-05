#!/usr/bin/env python3
"""Remove samples listed as candidates in a corpus quality-audit report."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    from .split_kulupu_lapo import SPLIT_RATIOS, verify
except ImportError:  # Direct execution: python scripts/remove_quality_outliers.py
    from split_kulupu_lapo import SPLIT_RATIOS, verify


MANIFEST_NAME = "split_manifest.json"


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=project / "quality_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = Path(__file__).resolve().parent.parent
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dataset_reports = report.get("datasets")
    if not isinstance(dataset_reports, list):
        print("error: audit report has no datasets list", file=sys.stderr)
        return 1

    staging = project / ".quality-outlier-staging"
    if staging.exists():
        print(f"error: staging directory already exists: {staging}", file=sys.stderr)
        return 1

    plans: list[dict[str, object]] = []
    for dataset_report in dataset_reports:
        if not isinstance(dataset_report, dict):
            print("error: malformed dataset report", file=sys.stderr)
            return 1
        root = Path(str(dataset_report.get("dataset", ""))).resolve()
        if project != root and project not in root.parents:
            print(f"error: dataset is outside the project: {root}", file=sys.stderr)
            return 1
        manifest_path = root / MANIFEST_NAME
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        entries = manifest.get("samples")
        candidates = dataset_report.get("candidates")
        if not isinstance(entries, list) or not isinstance(candidates, list):
            print(f"error: malformed report or manifest for {root}", file=sys.stderr)
            return 1

        entries_by_location = {
            (str(entry["split"]), str(entry["id"])): entry
            for entry in entries
            if isinstance(entry, dict)
        }
        removals: list[dict[str, object]] = []
        directories: list[tuple[Path, Path]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                print("error: malformed audit candidate", file=sys.stderr)
                return 1
            relative = Path(str(candidate.get("path", "")))
            if len(relative.parts) != 3 or relative.suffix != ".txt":
                print(f"error: invalid candidate path: {relative}", file=sys.stderr)
                return 1
            split, sample_id, filename = relative.parts
            if split not in SPLIT_RATIOS or filename != f"{sample_id}.txt":
                print(f"error: invalid candidate location: {relative}", file=sys.stderr)
                return 1
            entry = entries_by_location.get((split, sample_id))
            if entry is None:
                print(f"error: candidate is absent from manifest: {relative}", file=sys.stderr)
                return 1
            source_directory = root / split / sample_id
            if not source_directory.is_dir():
                print(f"error: missing candidate directory: {source_directory}", file=sys.stderr)
                return 1
            target_directory = staging / root.name / split / sample_id
            directories.append((source_directory, target_directory))
            removals.append(
                {
                    **entry,
                    "quality_reasons": candidate.get("reasons", []),
                    "word_runs": candidate.get("word_runs"),
                    "unfamiliar_words": candidate.get("unfamiliar_words"),
                    "unknown_share": candidate.get("unknown_share"),
                    "non_latin_share": candidate.get("non_latin_share"),
                    "mark_share": candidate.get("mark_share"),
                }
            )

        plans.append(
            {
                "root": root,
                "manifest_path": manifest_path,
                "manifest_text": manifest_text,
                "manifest": manifest,
                "entries": entries,
                "directories": directories,
                "removals": removals,
            }
        )

    total_removals = sum(len(plan["removals"]) for plan in plans)
    if total_removals == 0:
        print("no quality outliers are listed in the report")
        return 0

    staging.mkdir()
    try:
        for plan in plans:
            directories = plan["directories"]
            assert isinstance(directories, list)
            for source, target in directories:
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)

        for plan in plans:
            root = plan["root"]
            manifest_path = plan["manifest_path"]
            manifest = dict(plan["manifest"])
            entries = plan["entries"]
            removals = plan["removals"]
            assert isinstance(root, Path)
            assert isinstance(manifest_path, Path)
            assert isinstance(entries, list)
            assert isinstance(removals, list)

            removed_locations = {
                (str(removal["split"]), str(removal["id"])) for removal in removals
            }
            remaining = [
                entry
                for entry in entries
                if (str(entry["split"]), str(entry["id"])) not in removed_locations
            ]
            manifest["samples"] = remaining
            manifest["counts"] = {
                split: sum(1 for entry in remaining if entry["split"] == split)
                for split in SPLIT_RATIOS
            }
            if "token_totals" in manifest:
                manifest["token_totals"] = {
                    split: sum(
                        int(entry.get("tokens", 0))
                        for entry in remaining
                        if entry["split"] == split
                    )
                    for split in SPLIT_RATIOS
                }
            statistics = dict(manifest.get("statistics", {}))
            statistics["quality_outliers_removed"] = (
                int(statistics.get("quality_outliers_removed", 0)) + len(removals)
            )
            statistics["samples_written"] = len(remaining)
            manifest["statistics"] = statistics
            previous_removals = list(manifest.get("quality_outlier_removals", []))
            manifest["quality_outlier_removals"] = previous_removals + removals
            manifest["quality_filter"] = {
                "report": report_path.name,
                "criteria": report.get("criteria"),
            }

            temporary = root / f".{MANIFEST_NAME}.tmp"
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            temporary.replace(manifest_path)

        all_errors: list[str] = []
        for plan in plans:
            root = plan["root"]
            manifest_path = plan["manifest_path"]
            assert isinstance(root, Path)
            assert isinstance(manifest_path, Path)
            all_errors.extend(verify(root, manifest_path))
        if all_errors:
            raise ValueError("; ".join(all_errors))
    except BaseException:
        for plan in plans:
            manifest_path = plan["manifest_path"]
            manifest_text = plan["manifest_text"]
            assert isinstance(manifest_path, Path)
            assert isinstance(manifest_text, str)
            manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
        for plan in plans:
            directories = plan["directories"]
            assert isinstance(directories, list)
            for source, target in directories:
                if target.exists():
                    target.replace(source)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    shutil.rmtree(staging)
    for plan in plans:
        root = plan["root"]
        removals = plan["removals"]
        assert isinstance(root, Path)
        assert isinstance(removals, list)
        print(f"removed {len(removals):,} quality outliers from {root.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
