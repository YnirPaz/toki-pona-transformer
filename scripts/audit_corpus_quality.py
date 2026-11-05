#!/usr/bin/env python3
"""Audit processed corpora for samples that tokenize pathologically."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

try:
    from .analyze_tokenization import analyze_text
    from .extract_kulupu_lapo import Quality, clean_and_filter, pathology_reasons
except ImportError:  # Direct execution: python scripts/audit_corpus_quality.py
    from analyze_tokenization import analyze_text
    from extract_kulupu_lapo import Quality, clean_and_filter, pathology_reasons


SHORT_MIN_WORD_RUNS = 5
SHORT_MIN_UNFAMILIAR_WORDS = 4
SHORT_MAX_UNKNOWN_SHARE = 0.50
SHORT_MIN_NON_LATIN_UNITS = 4


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "quality_audit.json",
        help="JSON report path",
    )
    return parser.parse_args()


def metrics(text: str) -> tuple[Quality, Counter[str]]:
    counts = analyze_text(text)
    tokens = max(0, counts.total - counts.eos)
    word_runs = counts.words + counts.spell
    unknown_share = counts.spell / word_runs if word_runs else 0.0
    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    private_use = [character for character in text if unicodedata.category(character) == "Co"]
    non_latin_letters = [
        character
        for character in letters
        if not unicodedata.name(character, "").startswith("LATIN ")
    ]
    script_units = len(letters) + len(private_use)
    non_latin_units = len(non_latin_letters) + len(private_use)
    non_latin_share = non_latin_units / script_units if script_units else 0.0
    mark_share = counts.marks / tokens if tokens else 0.0
    return (
        Quality(
            tokens=tokens,
            word_runs=word_runs,
            unknown_share=unknown_share,
            script_units=script_units,
            non_latin_units=non_latin_units,
            non_latin_share=non_latin_share,
            mark_share=mark_share,
        ),
        counts.unfamiliar_words,
    )


def reasons_for_sample(
    sample: Quality,
    unfamiliar_count: int,
    *,
    include_short_gates: bool,
) -> list[str]:
    # This is the same document-level gate used before kulupu-lapo's optional
    # paragraph salvage. The extra short gates are only meaningful for a
    # sentence corpus, whose samples cannot reach the document minima.
    reasons = list(pathology_reasons(sample))
    if (
        include_short_gates
        and
        sample.word_runs < 32
        and sample.word_runs >= SHORT_MIN_WORD_RUNS
        and unfamiliar_count >= SHORT_MIN_UNFAMILIAR_WORDS
        and sample.unknown_share >= SHORT_MAX_UNKNOWN_SHARE
    ):
        reasons.append(
            f"short sample with {sample.unknown_share:.1%} unfamiliar words"
        )
    if (
        include_short_gates
        and
        sample.script_units < 16
        and sample.non_latin_units >= SHORT_MIN_NON_LATIN_UNITS
        and sample.non_latin_share >= 0.10
    ):
        reasons.append(f"short sample with {sample.non_latin_share:.1%} non-Latin")
    return reasons


def audit(dataset: Path) -> dict[str, object]:
    files = sorted(dataset.rglob("*.txt"), key=lambda path: path.as_posix().casefold())
    if not files:
        raise ValueError(f"no .txt files found under {dataset}")

    include_short_gates = "tatoeba" in dataset.name.casefold()
    candidates: list[dict[str, object]] = []
    share_buckets = Counter()
    aggregate_unfamiliar = Counter()
    total_word_runs = total_unfamiliar = total_tokens = 0
    for number, path in enumerate(files, start=1):
        text = path.read_text(encoding="utf-8")
        sample, unfamiliar = metrics(text)
        unfamiliar_count = sum(unfamiliar.values())
        total_word_runs += sample.word_runs
        total_unfamiliar += unfamiliar_count
        total_tokens += sample.tokens
        aggregate_unfamiliar.update(unfamiliar)

        for threshold in (0.35, 0.50, 0.67, 0.80, 1.00):
            if sample.unknown_share >= threshold:
                share_buckets[f">={threshold:.2f}"] += 1

        reasons = reasons_for_sample(
            sample,
            unfamiliar_count,
            include_short_gates=include_short_gates,
        )
        if reasons:
            preview = re.sub(r"\s+", " ", text).strip()[:300]
            kulupu_result = clean_and_filter(text)
            if kulupu_result.text is None:
                kulupu_action = "reject"
                salvage_tokens = 0
            elif kulupu_result.salvaged:
                kulupu_action = "salvage"
                salvage_tokens = analyze_text(kulupu_result.text).total - 1
            else:
                kulupu_action = "keep"
                salvage_tokens = sample.tokens
            candidates.append(
                {
                    "path": path.relative_to(dataset).as_posix(),
                    "reasons": reasons,
                    "tokens": sample.tokens,
                    "word_runs": sample.word_runs,
                    "unfamiliar_words": unfamiliar_count,
                    "unknown_share": sample.unknown_share,
                    "non_latin_share": sample.non_latin_share,
                    "mark_share": sample.mark_share,
                    "top_unfamiliar": unfamiliar.most_common(12),
                    "kulupu_lapo_action": kulupu_action,
                    "salvage_tokens": salvage_tokens,
                    "preview": preview,
                }
            )
        if number % 5000 == 0:
            print(f"{dataset.name}: audited {number:,}/{len(files):,}", flush=True)

    candidates.sort(
        key=lambda item: (
            float(item["unknown_share"]),
            int(item["unfamiliar_words"]),
            int(item["word_runs"]),
        ),
        reverse=True,
    )
    return {
        "dataset": str(dataset),
        "short_sample_gates_enabled": include_short_gates,
        "samples": len(files),
        "total_tokens_without_eos": total_tokens,
        "word_runs": total_word_runs,
        "unfamiliar_word_runs": total_unfamiliar,
        "unfamiliar_share": total_unfamiliar / total_word_runs,
        "sample_unknown_share_buckets": dict(share_buckets),
        "top_unfamiliar_words": aggregate_unfamiliar.most_common(50),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> int:
    args = parse_args()
    reports = []
    try:
        for dataset in args.datasets:
            reports.append(audit(dataset.resolve()))
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    payload = {
        "criteria": {
            "long_or_medium_samples": (
                "kulupu-lapo document gates: at least 128 word runs and >=35% "
                "unfamiliar, at least 64 script units and >=10% non-Latin, "
                "or at least 512 tokens and >=55% marks"
            ),
            "short_samples": (
                "Tatoeba only: "
                "at least 5 word runs, 4 unfamiliar runs, and >=50% unfamiliar; "
                "or at least 4 non-Latin units and >=10% non-Latin"
            ),
        },
        "datasets": reports,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for report in reports:
        print(
            f"{Path(str(report['dataset'])).name}: "
            f"{report['candidate_count']:,}/{report['samples']:,} candidates, "
            f"{report['unfamiliar_share']:.2%} unfamiliar word runs overall"
        )
    print(f"report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
