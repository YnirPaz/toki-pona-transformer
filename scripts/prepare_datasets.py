"""Recreate the historical three-corpus training set and tokenize it.

The small, versioned TSV files in dataset_locks/ record the original document
selection and split assignment. They also make changes to upstream text or to
our cleaner visible: each generated text file must match its recorded SHA-256.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unicodedata
import urllib.request
import zipfile
from array import array
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from scripts.build_tatoeba import clean_text
from scripts.build_wikipesija import clean_article_text
from scripts.extract_kulupu_lapo import clean_and_filter, extract_body_bytes
from tokenizer import TokiPonaTokenizer


PROJECT = Path(__file__).resolve().parents[1]
SPLITS = ("training", "validation", "evaluation")
POKI_REVISION = "275f55ef1c349880bf48c769e7241d4c6ed7fd6e"
TATOEBA_REVISION = "0b2b008f4da75cf02c9bf4b0b4f2aaa292689ca2"
WIKIPESIJA_REVISION = "19dd7d1abf276e8d083f073d9e6700d536f38128"


@dataclass(frozen=True)
class SourceFile:
    relative: str
    url: str
    sha256: str


SOURCES = (
    SourceFile(
        "kulupu-lapo.zip",
        f"https://codeload.github.com/kulupu-lapo/poki/zip/{POKI_REVISION}",
        "ecb6ef93a021c2e2c060f18deacc438a108a5ce3fca72aedf05f645a383f0634",
    ),
    *(
        SourceFile(
            f"{dataset}/data/{split}-00000-of-00001.parquet",
            f"https://huggingface.co/datasets/NetherQuartz/{dataset}/resolve/{revision}/data/"
            f"{split}-00000-of-00001.parquet?download=true",
            {
                ("tatoeba-tokipona", "train"): "1fcc1c2c495dfac36a0686d719ae6a82dd7ae959a34cc09af15ee3d252bb0253",
                ("tatoeba-tokipona", "validation"): "dc9536286d04be3f4f00c240c3c169eb7b2af1fc51fe7732bbda6fe7bac8061d",
                ("wikipesija", "train"): "5c0b9d083123623d147d76b0d300707cddc80dedac0d1435ac9d23d2f346a47c",
                ("wikipesija", "validation"): "b27e4d90dac4a9402187f3291ab4152ed66a1751638154a23402a12073bcc4b8",
            }[(dataset, split)],
        )
        for dataset, revision in (
            ("tatoeba-tokipona", TATOEBA_REVISION),
            ("wikipesija", WIKIPESIJA_REVISION),
        )
        for split in ("train", "validation")
    ),
)


@dataclass(frozen=True)
class LockedSample:
    split: str
    sample_id: str
    source: str
    sha256: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_lock(path: Path) -> list[LockedSample]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["split", "id", "source", "sha256"]:
            raise ValueError(f"unexpected lock header: {path}")
        samples = [
            LockedSample(row["split"], row["id"], row["source"], row["sha256"])
            for row in reader
        ]
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        if (
            sample.split not in SPLITS
            or not sample.sample_id.isdecimal()
            or not sample.source
            or len(sample.sha256) != 64
        ):
            raise ValueError(f"invalid lock row in {path}: {sample!r}")
        identity = (sample.split, sample.sample_id)
        if identity in seen:
            raise ValueError(f"repeated sample in {path}: {identity}")
        seen.add(identity)
    if not samples:
        raise ValueError(f"empty dataset lock: {path}")
    return samples


def download_sources(raw_dir: Path) -> None:
    for source in SOURCES:
        destination = raw_dir / source.relative
        if destination.is_file():
            if sha256(destination.read_bytes()) != source.sha256:
                raise ValueError(f"raw source has the wrong SHA-256: {destination}")
            print(f"using {destination}", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {source.url}", flush=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                request = urllib.request.Request(
                    source.url, headers={"User-Agent": "TokiPonaTransformer/1.0"}
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    shutil.copyfileobj(response, temporary)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        if sha256(temporary_path.read_bytes()) != source.sha256:
            temporary_path.unlink(missing_ok=True)
            raise ValueError(f"download has the wrong SHA-256: {source.url}")
        temporary_path.replace(destination)


def read_parquet_rows(directory: Path, id_column: str, columns: tuple[str, ...]) -> dict[int, dict[str, object]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("PyArrow is required; install requirements-data.txt") from error
    files = sorted((directory / "data").glob("*.parquet"))
    if len(files) != 2:
        raise ValueError(f"expected two Parquet files under {directory / 'data'}")
    result: dict[int, dict[str, object]] = {}
    for file in files:
        for batch in parquet.ParquetFile(file).iter_batches(columns=list(columns)):
            data = batch.to_pydict()
            for values in zip(*(data[column] for column in columns), strict=True):
                row = dict(zip(columns, values, strict=True))
                source_id = int(row[id_column])
                previous = result.setdefault(source_id, row)
                if previous != row:
                    # Tatoeba may have several translations of one sentence.
                    if id_column == "target_id" and previous["tok"] == row["tok"]:
                        continue
                    raise ValueError(f"conflicting rows for {id_column}={source_id}")
    return result


def load_poki(archive_path: Path) -> tuple[zipfile.ZipFile, dict[str, str]]:
    archive = zipfile.ZipFile(archive_path)
    names = {
        unicodedata.normalize("NFC", name.split("/plaintext/", 1)[1]): name
        for name in archive.namelist()
        if "/plaintext/" in name and name.endswith(".md")
    }
    if not names:
        archive.close()
        raise ValueError(f"no poki Markdown files in {archive_path}")
    return archive, names


def build_one(
    name: str,
    samples: list[LockedSample],
    raw_dir: Path,
    output: Path,
    tokenizer: TokiPonaTokenizer,
) -> dict[str, object]:
    output.mkdir()
    archive = None
    names: dict[str, str] = {}
    if name == "kulupu-lapo":
        archive, names = load_poki(raw_dir / "kulupu-lapo.zip")
        rows: dict[int, dict[str, object]] = {}
        source_name = f"kulupu-lapo/poki@{POKI_REVISION}"
    elif name == "tatoeba":
        rows = read_parquet_rows(
            raw_dir / "tatoeba-tokipona", "target_id", ("target_id", "tok")
        )
        source_name = f"NetherQuartz/tatoeba-tokipona@{TATOEBA_REVISION}"
    else:
        rows = read_parquet_rows(
            raw_dir / "wikipesija", "id", ("id", "url", "title", "text")
        )
        source_name = f"NetherQuartz/wikipesija@{WIKIPESIJA_REVISION}"

    manifest_samples: list[dict[str, object]] = []
    counts = {split: 0 for split in SPLITS}
    token_totals = {split: 0 for split in SPLITS}
    streams = {}
    try:
        for split in SPLITS:
            (output / split).mkdir()
            streams[split] = (output / f"{split}.uint16le").open("wb")
        with (output / "metadata.jsonl").open("w", encoding="utf-8", newline="\n") as metadata:
            for position, sample in enumerate(samples, start=1):
                if name == "kulupu-lapo":
                    assert archive is not None
                    member = names.get(unicodedata.normalize("NFC", sample.source))
                    if member is None:
                        raise ValueError(f"missing poki source: {sample.source}")
                    source_bytes = archive.read(member)
                    cleaned = clean_and_filter(extract_body_bytes(source_bytes))
                    if cleaned.text is None:
                        raise ValueError(f"selected poki source was rejected: {sample.source}")
                    text = cleaned.text
                    source_url = (
                        f"https://github.com/kulupu-lapo/poki/blob/{POKI_REVISION}/"
                        f"plaintext/{sample.source}"
                    )
                    details = {"source_path": sample.source, "source_url": source_url}
                elif name == "tatoeba":
                    row = rows.get(int(sample.source))
                    if row is None:
                        raise ValueError(f"missing Tatoeba sentence: {sample.source}")
                    text = clean_text(str(row["tok"] or "")) + "\n"
                    details = {
                        "tatoeba_id": int(sample.source),
                        "source_url": f"https://tatoeba.org/en/sentences/show/{sample.source}",
                        "license": "CC BY 2.0",
                    }
                else:
                    row = rows.get(int(sample.source))
                    if row is None:
                        raise ValueError(f"missing Wikipesija article: {sample.source}")
                    text = clean_article_text(str(row["text"] or "")) + "\n"
                    details = {
                        "wikipesija_id": int(sample.source),
                        "source_url": row["url"],
                        "title": row["title"],
                        "license": "CC BY-SA 3.0",
                    }
                content = text.encode("utf-8")
                if sha256(content) != sample.sha256:
                    raise ValueError(
                        f"source or cleaning changed: {name}/{sample.split}/{sample.sample_id} "
                        f"({sample.source})"
                    )
                text_path = output / sample.split / f"{sample.sample_id}.txt"
                text_path.write_bytes(content)
                ids = tokenizer.encode(text)
                token_array = array("H", ids)
                if sys.byteorder != "little":
                    token_array.byteswap()
                token_array.tofile(streams[sample.split])
                entry: dict[str, object] = {
                    "split": sample.split,
                    "id": sample.sample_id,
                    "text": f"{sample.split}/{sample.sample_id}.txt",
                    "source": sample.source,
                    "text_sha256": sample.sha256,
                    "token_offset": token_totals[sample.split],
                    "token_count": len(ids),
                    **details,
                }
                metadata.write(json.dumps(entry, ensure_ascii=False) + "\n")
                manifest_samples.append(entry)
                counts[sample.split] += 1
                token_totals[sample.split] += len(ids)
                if position % 10000 == 0:
                    print(f"{name}: prepared {position:,}/{len(samples):,}", flush=True)
    finally:
        for stream in streams.values():
            stream.close()
        if archive is not None:
            archive.close()

    manifest: dict[str, object] = {
        "version": 1,
        "source": source_name,
        "selection": f"dataset_locks/{name}.tsv",
        "splits": counts,
        "token_totals": token_totals,
        "token_format": "unsigned 16-bit little-endian; concatenated documents in manifest order",
        "tokenizer": "../tokenizer.json",
        "samples": manifest_samples,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"{name}: {len(samples):,} documents, {sum(token_totals.values()):,} tokens", flush=True)
    return manifest


def verify_one(root: Path, lock: list[LockedSample], tokenizer: TokiPonaTokenizer) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["samples"]
    if len(entries) != len(lock):
        raise ValueError(f"wrong sample count in {root}")
    with (root / "metadata.jsonl").open(encoding="utf-8") as metadata:
        records = [json.loads(line) for line in metadata]
    if records != entries:
        raise ValueError(f"metadata differs from manifest in {root}")
    offsets = {split: 0 for split in SPLITS}
    counts = {split: 0 for split in SPLITS}
    token_files = {split: (root / f"{split}.uint16le").read_bytes() for split in SPLITS}
    for entry, expected in zip(entries, lock, strict=True):
        if (entry["split"], entry["id"], entry["source"], entry["text_sha256"]) != (
            expected.split, expected.sample_id, expected.source, expected.sha256
        ):
            raise ValueError(f"manifest differs from lock in {root}")
        path = root / entry["text"]
        content = path.read_bytes()
        if sha256(content) != expected.sha256:
            raise ValueError(f"text hash differs from lock: {path}")
        ids = tokenizer.encode(content.decode("utf-8"))
        split = expected.split
        if entry["token_offset"] != offsets[split] or entry["token_count"] != len(ids):
            raise ValueError(f"token index differs: {path}")
        tokens = array("H", ids)
        if sys.byteorder != "little":
            tokens.byteswap()
        start = 2 * offsets[split]
        if token_files[split][start : start + 2 * len(ids)] != tokens.tobytes():
            raise ValueError(f"token stream differs: {path}")
        offsets[split] += len(ids)
        counts[split] += 1
    if counts != manifest["splits"] or offsets != manifest["token_totals"]:
        raise ValueError(f"manifest totals differ: {root}")
    for split in SPLITS:
        if len(token_files[split]) != 2 * offsets[split]:
            raise ValueError(f"token stream has extra data: {root}/{split}")
        actual = list((root / split).glob("*.txt"))
        if len(actual) != counts[split]:
            raise ValueError(f"unexpected text files: {root}/{split}")


def prepare(raw_dir: Path, output: Path, locks_dir: Path, *, check: bool, rebuild: bool) -> None:
    locks = {name: read_lock(locks_dir / f"{name}.tsv") for name in (
        "kulupu-lapo", "tatoeba", "wikipesija"
    )}
    tokenizer = TokiPonaTokenizer()
    if check or (output.exists() and not rebuild):
        TokiPonaTokenizer.load(output / "tokenizer.json")
        for name, lock in locks.items():
            verify_one(output / name, lock, tokenizer)
        print("verified all three corpora and token streams", flush=True)
        return
    download_sources(raw_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".prepare-{uuid4().hex}"
    staging.mkdir()
    try:
        staged = staging / "processed"
        staged.mkdir()
        tokenizer.save(staged / "tokenizer.json")
        for name, lock in locks.items():
            build_one(name, lock, raw_dir, staged / name, tokenizer)
            verify_one(staged / name, lock, tokenizer)
        if output.exists():
            backup = output.with_name(f".{output.name}-backup")
            if backup.exists():
                raise FileExistsError(f"refusing to overwrite backup: {backup}")
            output.replace(backup)
            try:
                staged.replace(output)
            except BaseException:
                backup.replace(output)
                raise
            shutil.rmtree(backup)
        else:
            staged.replace(output)
    finally:
        shutil.rmtree(staging)
    print(f"ready for training: {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=PROJECT / "data" / "processed")
    parser.add_argument("--locks-dir", type=Path, default=PROJECT / "dataset_locks")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="verify existing texts and token arrays")
    actions.add_argument("--rebuild", action="store_true", help="replace existing output after a full successful build")
    arguments = parser.parse_args()
    try:
        prepare(
            arguments.raw_dir.resolve(), arguments.output.resolve(),
            arguments.locks_dir.resolve(), check=arguments.check, rebuild=arguments.rebuild,
        )
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
