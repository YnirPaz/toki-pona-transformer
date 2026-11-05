#!/usr/bin/env python3
"""Build Wikipesija in the project's split corpus format.

Articles are deduplicated internally and against kulupu-lapo. Any retained
Tatoeba sample whose normalized word sequence occurs inside a Wikipesija
article is removed from the processed Tatoeba corpus, preserving the longer
article as the canonical copy.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from .analyze_tokenization import analyze_text
    from .build_tatoeba import word_key
    from .split_kulupu_lapo import SPLIT_RATIOS, verify
except ImportError:  # Direct execution: python scripts/build_wikipesija.py
    from analyze_tokenization import analyze_text
    from build_tatoeba import word_key
    from split_kulupu_lapo import SPLIT_RATIOS, verify


DEFAULT_SEED = 20261024
MANIFEST_NAME = "split_manifest.json"
END = object()


@dataclass(frozen=True)
class Article:
    wikipesija_id: int
    url: str
    title: str
    text: str
    key: tuple[str, ...]
    tokens: int


@dataclass(frozen=True)
class CorpusSample:
    entry: dict[str, object]
    directory: Path
    text: str
    key: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=project / "wikipesija",
        help="downloaded Wikipesija dataset directory",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=project / "wikipesija-processed",
        help="destination dataset directory",
    )
    parser.add_argument(
        "--kulupu-lapo",
        type=Path,
        default=project / "kulupu-lapo",
        help="long-form reference corpus",
    )
    parser.add_argument(
        "--tatoeba",
        type=Path,
        default=project / "tatoeba-tokipona-processed",
        help="processed Tatoeba corpus to check and prune",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify all processed corpora without changing them",
    )
    return parser.parse_args()


def clean_article_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def read_articles(source: Path) -> tuple[list[Article], dict[str, int]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit(
            "error: PyArrow is required to read the Wikipesija Parquet files "
            "(install it with: python -m pip install pyarrow)"
        ) from error

    parquet_files = sorted((source / "data").glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"no Parquet files found under {source / 'data'}")

    by_key: dict[tuple[str, ...], Article] = {}
    input_rows = empty_rows = 0
    for parquet_file in parquet_files:
        parquet_source = parquet.ParquetFile(parquet_file)
        for batch in parquet_source.iter_batches(columns=["id", "url", "title", "text"]):
            data = batch.to_pydict()
            rows = zip(
                data["id"], data["url"], data["title"], data["text"], strict=True
            )
            for wikipesija_id, url, title, raw_text in rows:
                input_rows += 1
                text = clean_article_text(raw_text or "")
                key = word_key(text)
                if not key:
                    empty_rows += 1
                    continue
                article = Article(
                    wikipesija_id=int(wikipesija_id),
                    url=str(url),
                    title=str(title),
                    text=text,
                    key=key,
                    tokens=analyze_text(text).total,
                )
                previous = by_key.get(key)
                if previous is None or article.wikipesija_id < previous.wikipesija_id:
                    by_key[key] = article

    articles = sorted(by_key.values(), key=lambda article: article.wikipesija_id)
    return articles, {
        "input_rows": input_rows,
        "empty_rows_removed": empty_rows,
        "within_wikipesija_duplicates_removed": input_rows - empty_rows - len(articles),
    }


def split_counts(sample_count: int) -> dict[str, int]:
    validation = round(sample_count * SPLIT_RATIOS["validation"])
    evaluation = round(sample_count * SPLIT_RATIOS["evaluation"])
    return {
        "training": sample_count - validation - evaluation,
        "validation": validation,
        "evaluation": evaluation,
    }


def find_occurrences(
    candidate_keys: set[tuple[str, ...]],
    documents: list[tuple[int, str]],
) -> dict[tuple[str, ...], set[int]]:
    """Map candidate word sequences to documents containing them."""
    trie: dict[object, dict] = {}
    for key in candidate_keys:
        node = trie
        for word in key:
            node = node.setdefault(word, {})
        node[END] = key

    matches: dict[tuple[str, ...], set[int]] = {}
    for document_id, text in documents:
        words = word_key(text)
        for start in range(len(words)):
            node = trie
            for word in words[start:]:
                child = node.get(word)
                if child is None:
                    break
                node = child
                match = node.get(END)
                if match is not None:
                    matches.setdefault(match, set()).add(document_id)
    return matches


def text_documents(root: Path) -> list[tuple[int, str]]:
    paths = sorted(root.rglob("*.txt"), key=lambda path: path.as_posix().casefold())
    if not paths:
        raise ValueError(f"no .txt samples found under {root}")
    return [
        (number, path.read_text(encoding="utf-8"))
        for number, path in enumerate(paths)
    ]


def load_processed_samples(root: Path) -> tuple[dict[str, object], list[CorpusSample]]:
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("samples")
    if not isinstance(entries, list):
        raise ValueError(f"manifest has no samples list: {manifest_path}")

    samples: list[CorpusSample] = []
    seen: set[tuple[str, ...]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("malformed processed-corpus manifest entry")
        split = entry.get("split")
        sample_id = entry.get("id")
        if split not in SPLIT_RATIOS or not isinstance(sample_id, str):
            raise ValueError(f"invalid processed-corpus manifest entry: {entry!r}")
        directory = root / split / sample_id
        text = (directory / f"{sample_id}.txt").read_text(encoding="utf-8")
        key = word_key(text)
        if not key or key in seen:
            raise ValueError(f"empty or duplicate processed sample: {directory}")
        seen.add(key)
        samples.append(CorpusSample(entry, directory, text, key))
    return manifest, samples


def allocate_articles(
    articles: list[Article], counts: dict[str, int], seed: int
) -> tuple[dict[str, list[Article]], dict[str, int]]:
    """Token-balance articles while preserving exact sample-count ratios."""
    rng = random.Random(seed)
    ordered = articles.copy()
    rng.shuffle(ordered)
    ordered.sort(key=lambda article: article.tokens, reverse=True)
    total_tokens = sum(article.tokens for article in articles)
    targets = {
        split: total_tokens * ratio for split, ratio in SPLIT_RATIOS.items()
    }
    groups: dict[str, list[Article]] = {split: [] for split in SPLIT_RATIOS}
    totals = {split: 0 for split in SPLIT_RATIOS}

    for article in ordered:
        eligible = [
            split for split in SPLIT_RATIOS if len(groups[split]) < counts[split]
        ]
        tie_breakers = {split: rng.random() for split in eligible}
        selected = min(
            eligible,
            key=lambda split: (
                (totals[split] + article.tokens) / targets[split],
                tie_breakers[split],
            ),
        )
        groups[selected].append(article)
        totals[selected] += article.tokens

    for split in SPLIT_RATIOS:
        rng.shuffle(groups[split])
    return groups, totals


def markdown_source(article: Article) -> str:
    title = json.dumps(article.title, ensure_ascii=False)
    url = json.dumps(article.url, ensure_ascii=False)
    return (
        "---\n"
        f"title: {title}\n"
        "description: null\n"
        "authors: null\n"
        "proofreaders: null\n"
        "date: null\n"
        "date-precision: null\n"
        "tags:\n"
        "  - Wikipesija\n"
        "license: CC BY-SA 3.0\n"
        "sources:\n"
        f"  - {url}\n"
        "archives: null\n"
        "preprocessing: extracted from the text field; whitespace normalized\n"
        "accessibility-notes: null\n"
        "notes: null\n"
        "---\n"
        f"{article.text}\n"
    )


def write_dataset(
    output: Path,
    groups: dict[str, list[Article]],
    counts: dict[str, int],
    token_totals: dict[str, int],
    seed: int,
    statistics: dict[str, int],
    overlap_records: list[dict[str, object]],
) -> None:
    width = max(4, len(str(max(counts.values()))))
    entries: list[dict[str, object]] = []
    output.mkdir(parents=True)
    for split, articles in groups.items():
        for number, article in enumerate(articles, start=1):
            sample_id = f"{number:0{width}d}"
            sample_dir = output / split / sample_id
            sample_dir.mkdir(parents=True)
            (sample_dir / f"{sample_id}.md").write_text(
                markdown_source(article), encoding="utf-8", newline="\n"
            )
            (sample_dir / f"{sample_id}.txt").write_text(
                article.text + "\n", encoding="utf-8", newline="\n"
            )
            entries.append(
                {
                    "split": split,
                    "id": sample_id,
                    "original_md": f"wikipesija/{article.wikipesija_id}.md",
                    "wikipesija_id": article.wikipesija_id,
                    "url": article.url,
                    "title": article.title,
                    "tokens": article.tokens,
                }
            )

    manifest = {
        "version": 2,
        "source": "NetherQuartz/wikipesija",
        "source_license": "CC BY-SA 3.0",
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "counts": counts,
        "token_totals": token_totals,
        "assignment_method": "token-balanced-greedy",
        "deduplication": {
            "comparison": "NFC casefolded Unicode word sequence",
            "long_form_reference": "kulupu-lapo",
            "tatoeba_policy": "remove shorter Tatoeba samples found in articles",
        },
        "statistics": statistics,
        "tatoeba_overlap_removals": overlap_records,
        "samples": entries,
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def prune_tatoeba(
    root: Path,
    manifest: dict[str, object],
    samples: list[CorpusSample],
    overlaps: dict[tuple[str, ...], set[int]],
) -> list[dict[str, object]]:
    removed = [sample for sample in samples if sample.key in overlaps]
    if not removed:
        return []

    staging = root.parent / ".tatoeba-wikipesija-overlap-staging"
    if staging.exists():
        raise ValueError(f"staging directory already exists: {staging}")
    staging.mkdir()
    manifest_path = root / MANIFEST_NAME
    old_manifest = manifest_path.read_text(encoding="utf-8")

    records: list[dict[str, object]] = []
    try:
        for sample in removed:
            split = str(sample.entry["split"])
            sample_id = str(sample.entry["id"])
            target = staging / split / sample_id
            target.parent.mkdir(parents=True, exist_ok=True)
            sample.directory.replace(target)
            records.append(
                {
                    "split": split,
                    "id": sample_id,
                    "tatoeba_id": sample.entry.get("tatoeba_id"),
                    "word_count": len(sample.key),
                    "wikipesija_ids": sorted(overlaps[sample.key]),
                }
            )

        removed_keys = set(overlaps)
        remaining_entries = [
            sample.entry for sample in samples if sample.key not in removed_keys
        ]
        new_manifest = dict(manifest)
        new_manifest["samples"] = remaining_entries
        new_manifest["counts"] = dict(
            Counter(str(entry["split"]) for entry in remaining_entries)
        )
        statistics = dict(new_manifest.get("statistics", {}))
        statistics["wikipesija_overlaps_removed"] = len(removed)
        statistics["samples_written"] = len(remaining_entries)
        new_manifest["statistics"] = statistics
        deduplication = dict(new_manifest.get("deduplication", {}))
        deduplication["additional_reference"] = "wikipesija"
        deduplication["additional_reference_scope"] = (
            "contiguous sequence anywhere in each article"
        )
        new_manifest["deduplication"] = deduplication
        new_manifest["wikipesija_overlap_removals"] = records

        temporary_manifest = root / f".{MANIFEST_NAME}.tmp"
        temporary_manifest.write_text(
            json.dumps(new_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_manifest.replace(manifest_path)
        errors = verify(root, manifest_path)
        if errors:
            raise ValueError("; ".join(errors))
    except BaseException:
        manifest_path.write_text(old_manifest, encoding="utf-8", newline="\n")
        for sample in removed:
            split = str(sample.entry["split"])
            sample_id = str(sample.entry["id"])
            staged = staging / split / sample_id
            if staged.exists():
                staged.replace(sample.directory)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    shutil.rmtree(staging)
    return records


def build(
    source: Path,
    output: Path,
    kulupu_lapo: Path,
    tatoeba: Path,
    seed: int,
) -> int:
    if output.exists():
        print(f"error: output already exists: {output}", file=sys.stderr)
        return 1

    articles, statistics = read_articles(source)
    kulupu_documents = text_documents(kulupu_lapo)
    article_matches = find_occurrences(
        {article.key for article in articles}, kulupu_documents
    )
    articles = [article for article in articles if article.key not in article_matches]
    statistics["kulupu_lapo_overlaps_removed"] = len(article_matches)

    tatoeba_manifest, tatoeba_samples = load_processed_samples(tatoeba)
    article_documents = [
        (article.wikipesija_id, article.text) for article in articles
    ]
    tatoeba_matches = find_occurrences(
        {sample.key for sample in tatoeba_samples}, article_documents
    )
    matched_samples = [
        sample for sample in tatoeba_samples if sample.key in tatoeba_matches
    ]
    matched_by_split = Counter(str(sample.entry["split"]) for sample in matched_samples)
    statistics["tatoeba_samples_found_in_articles"] = len(matched_samples)
    statistics["tatoeba_overlap_tokens"] = sum(
        int(sample.entry.get("tokens", analyze_text(sample.text).total))
        for sample in matched_samples
    )
    statistics["samples_written"] = len(articles)

    counts = split_counts(len(articles))
    groups, token_totals = allocate_articles(articles, counts, seed)
    provisional_records = [
        {
            "split": str(sample.entry["split"]),
            "id": str(sample.entry["id"]),
            "tatoeba_id": sample.entry.get("tatoeba_id"),
            "word_count": len(sample.key),
            "wikipesija_ids": sorted(tatoeba_matches[sample.key]),
        }
        for sample in matched_samples
    ]
    write_dataset(
        output,
        groups,
        counts,
        token_totals,
        seed,
        statistics,
        provisional_records,
    )
    errors = verify(output, output / MANIFEST_NAME)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    records = prune_tatoeba(
        tatoeba, tatoeba_manifest, tatoeba_samples, tatoeba_matches
    )
    if records != provisional_records:
        print("error: overlap audit changed during pruning", file=sys.stderr)
        return 1

    remaining_tatoeba = [
        sample for sample in tatoeba_samples if sample.key not in tatoeba_matches
    ]
    remaining_matches = find_occurrences(
        {sample.key for sample in remaining_tatoeba}, article_documents
    )
    if remaining_matches:
        print(
            f"error: {len(remaining_matches)} Tatoeba overlaps remain after pruning",
            file=sys.stderr,
        )
        return 1

    print(
        f"built {len(articles):,} articles: "
        + ", ".join(f"{split}={counts[split]:,}" for split in SPLIT_RATIOS)
    )
    print(
        "token totals: "
        + ", ".join(f"{split}={token_totals[split]:,}" for split in SPLIT_RATIOS)
    )
    print(
        f"removed {statistics['within_wikipesija_duplicates_removed']:,} internal "
        f"duplicates and {statistics['kulupu_lapo_overlaps_removed']:,} "
        "kulupu-lapo overlaps"
    )
    print(
        f"found and removed {len(records):,} Tatoeba samples in articles "
        f"({', '.join(f'{split}={matched_by_split[split]:,}' for split in SPLIT_RATIOS)})"
    )
    return 0


def check(output: Path, kulupu_lapo: Path, tatoeba: Path) -> int:
    errors = verify(output, output / MANIFEST_NAME)
    errors.extend(verify(tatoeba, tatoeba / MANIFEST_NAME))
    _, wiki_samples = load_processed_samples(output)
    _, tatoeba_samples = load_processed_samples(tatoeba)
    wiki_documents = [
        (number, sample.text) for number, sample in enumerate(wiki_samples)
    ]
    kulupu_documents = text_documents(kulupu_lapo)

    wiki_keys = {sample.key for sample in wiki_samples}
    tatoeba_keys = {sample.key for sample in tatoeba_samples}
    wiki_lapo = find_occurrences(wiki_keys, kulupu_documents)
    tatoeba_lapo = find_occurrences(tatoeba_keys, kulupu_documents)
    tatoeba_wiki = find_occurrences(tatoeba_keys, wiki_documents)
    if wiki_lapo:
        errors.append(f"{len(wiki_lapo)} Wikipesija samples overlap kulupu-lapo")
    if tatoeba_lapo:
        errors.append(f"{len(tatoeba_lapo)} Tatoeba samples overlap kulupu-lapo")
    if tatoeba_wiki:
        errors.append(f"{len(tatoeba_wiki)} Tatoeba samples overlap Wikipesija")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        f"verified {len(wiki_samples):,} Wikipesija and "
        f"{len(tatoeba_samples):,} Tatoeba samples with no cross-corpus overlap"
    )
    return 0


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    kulupu_lapo = args.kulupu_lapo.resolve()
    tatoeba = args.tatoeba.resolve()
    if args.check:
        return check(output, kulupu_lapo, tatoeba)
    return build(source, output, kulupu_lapo, tatoeba, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
