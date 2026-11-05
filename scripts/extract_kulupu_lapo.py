#!/usr/bin/env python3
"""Build clean Toki Pona text samples from the kulupu-lapo Markdown corpus.

The Markdown files remain the lossless source layer. This script writes a
cleaned ``.txt`` file beside each usable source, removes stale ``.txt`` outputs
for rejected sources, and never changes the original ``.md`` files.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:
    from .analyze_tokenization import analyze_text
except ImportError:  # Direct execution: python scripts/extract_kulupu_lapo.py
    from analyze_tokenization import analyze_text


UTF8_BOM = b"\xef\xbb\xbf"
MIN_SALVAGED_TOKENS = 256
MAX_NON_LATIN_SHARE = 0.10
MAX_UNKNOWN_WORD_SHARE = 0.35
MAX_MARK_SHARE = 0.55

BLOCK_TAGS = (
    "address|article|aside|blockquote|br|dd|details|div|dl|dt|figcaption|"
    "figure|footer|form|h[1-6]|header|hr|li|main|nav|ol|p|pre|section|"
    "summary|table|tbody|td|tfoot|th|thead|tr|ul"
)


@dataclass(frozen=True)
class Quality:
    tokens: int
    word_runs: int
    unknown_share: float
    script_units: int
    non_latin_units: int
    non_latin_share: float
    mark_share: float


@dataclass(frozen=True)
class CleanResult:
    text: str | None
    reasons: tuple[str, ...]
    salvaged: bool


def _line_without_ending(line: bytes) -> bytes:
    return line.rstrip(b"\r\n")


def extract_body(source: Path) -> str:
    """Return the UTF-8 document body after its YAML front matter."""
    return extract_body_bytes(source.read_bytes())


def extract_body_bytes(data: bytes) -> str:
    """Read a source directly from an archive, without extracting long paths."""
    if data.startswith(UTF8_BOM):
        data = data[len(UTF8_BOM) :]

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"not valid UTF-8: {error}") from error

    lines = data.splitlines(keepends=True)
    if not lines or _line_without_ending(lines[0]).strip() != b"---":
        raise ValueError("missing opening YAML front matter delimiter")

    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if _line_without_ending(line).strip() == b"---"
        ),
        None,
    )
    if closing_index is None:
        raise ValueError("missing closing YAML front matter delimiter")

    body_lines = text.splitlines(keepends=True)[closing_index + 1 :]
    if body_lines and not body_lines[0].strip():
        body_lines = body_lines[1:]
    return "".join(body_lines)


def markdown_to_text(markdown: str) -> str:
    """Conservatively retain visible prose while removing serialization noise."""
    text = unicodedata.normalize("NFC", markdown).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u200b", "")

    # Content that is never useful as natural-language training text.
    text = re.sub(r"(?is)<!--.*?-->", "\n", text)
    text = re.sub(r"(?is)<(?:style|script|noscript|svg)\b.*?</(?:style|script|noscript|svg)\s*>", "\n", text)
    text = re.sub(
        r"(?is)<(?P<tag>span|div)\b(?=[^>]*\bid\s*=\s*['\"]len['\"])[^>]*>.*?</(?P=tag)\s*>",
        "\n",
        text,
    )
    text = re.sub(r"(?is)<img\b[^>]*>", "\n", text)

    # Keep human-written image alt text and link labels, but discard targets.
    text = re.sub(r'(?m)^\s*\{\{\{sitelen\s+"[^"]*"\s+"([^"]*)"\s+"[^"]*"\}\}\}\s*$', r"\1", text)
    text = re.sub(r"!\[\[([^\]]+)\]\]\([^)]*\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", text)
    text = re.sub(r"(?im)^\s*\[[^\]]+\]:\s*(?:https?|ftp)://\S+\s*$", "", text)
    text = re.sub(r"(?i)\b(?:https?|ftp)://[^\s<>]*/\s+[^\s<>]+", " ", text)
    text = re.sub(r"(?i)\b(?:https?|ftp)://\s*[^\s<>]+", " ", text)
    text = re.sub(r"(?i)\bdata:image/[^\s<>]+", " ", text)

    # Preserve escaped literal angle brackets while stripping actual HTML.
    text = text.replace(r"\<", "\ufff0").replace(r"\>", "\ufff1")
    text = re.sub(rf"(?is)</?(?:{BLOCK_TAGS})\b[^>]*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", "", text)
    text = text.replace("\ufff0", "<").replace("\ufff1", ">")
    text = re.sub(r"(?is)</?(?:abbr|b|del|em|i|ins|s|span|strong)\b[^>]*>", "", text)
    text = html.unescape(text)
    text = re.sub(r"(?i)(?:\s*DATA CORRUPTED\s*)+", " ", text)

    # Long base64/hex/data runs can also appear as visible decorative content.
    text = re.sub(r"(?<![\w])[A-Za-z0-9+/=]{128,}(?![\w])", " ", text)

    cleaned_lines: list[str] = []
    for original_line in text.split("\n"):
        line = original_line
        if re.match(r"^\s*(?:```|~~~)", line):
            cleaned_lines.append("")
            continue

        # Remove data-dump lines that survived because they had indentation or
        # small fragments of markup around them.
        compact = re.sub(r"\s+", "", line)
        encoded = re.sub(r"[^0-9A-Fa-f+/=]", "", compact)
        if len(compact) >= 128 and len(encoded) / len(compact) >= 0.80:
            cleaned_lines.append("")
            continue

        line = re.sub(r"^\s{0,3}>\s?", "", line)
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[-+*]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)
        if re.fullmatch(r"\s*(?:[-*_]\s*){3,}", line):
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*", line):
            cleaned_lines.append("")
            continue

        line = "".join(
            " " if unicodedata.category(character) == "Zs" else character
            for character in line
        )
        line = re.sub(r"\\([\[\]])", r"\1", line)
        line = re.sub(r"^\s*\[\^?[^\]]+\]:\s*", "", line)
        line = re.sub(r"\[\^[^\]]+\]", "", line)
        line = re.sub(r"\[\d+\]", "", line)
        line = re.sub(r"\[([^\]]+)\]", r"\1", line)
        line = line.replace("|", " ")
        line = re.sub(r"(?<!\\)(?:\*\*|__|~~|`)", "", line)
        line = re.sub(r"(?<!\\)[*_]", "", line)
        line = re.sub(r"\\([\\`*_{}\[\]()#+.!<>-])", r"\1", line)
        line = re.sub(r"\\\s*$", "", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return f"{text}\n" if text else ""


def quality(text: str) -> Quality:
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
    return Quality(
        tokens=tokens,
        word_runs=word_runs,
        unknown_share=unknown_share,
        script_units=script_units,
        non_latin_units=non_latin_units,
        non_latin_share=non_latin_share,
        mark_share=mark_share,
    )


def pathology_reasons(metrics: Quality, *, block: bool = False) -> tuple[str, ...]:
    reasons: list[str] = []
    minimum_script_units = 16 if block else 64
    minimum_word_runs = 32 if block else 128
    minimum_tokens = 128 if block else 512

    if (
        metrics.non_latin_units >= minimum_script_units
        and metrics.non_latin_share >= MAX_NON_LATIN_SHARE
    ):
        reasons.append(f"{metrics.non_latin_share:.1%} non-Latin")
    if metrics.word_runs >= minimum_word_runs and metrics.unknown_share >= MAX_UNKNOWN_WORD_SHARE:
        reasons.append(f"{metrics.unknown_share:.1%} unfamiliar words")
    if metrics.tokens >= minimum_tokens and metrics.mark_share >= MAX_MARK_SHARE:
        reasons.append(f"{metrics.mark_share:.1%} punctuation/formatting")
    return tuple(reasons)


def _salvage_long_clean_sections(text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    retained: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        section = "\n\n".join(current).strip()
        if quality(section).tokens >= MIN_SALVAGED_TOKENS:
            retained.append(section)
        current.clear()

    for block in blocks:
        if pathology_reasons(quality(block), block=True):
            flush()
        else:
            current.append(block)
    flush()
    return retained


def clean_and_filter(markdown_body: str) -> CleanResult:
    text = markdown_to_text(markdown_body)
    if not text.strip():
        return CleanResult(None, ("empty after cleanup",), False)

    reasons = pathology_reasons(quality(text))
    if not reasons:
        return CleanResult(text, (), False)

    sections = _salvage_long_clean_sections(text)
    if not sections:
        return CleanResult(None, reasons, False)

    salvaged = "\n\n".join(sections).strip() + "\n"
    # A retained section must itself pass the document-level gates.
    remaining_reasons = pathology_reasons(quality(salvaged), block=True)
    if remaining_reasons:
        return CleanResult(None, reasons + remaining_reasons, False)
    return CleanResult(salvaged, reasons, True)


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
        "--check",
        action="store_true",
        help="verify outputs and rejection decisions without writing or deleting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        print(f"error: dataset directory does not exist: {dataset}", file=sys.stderr)
        return 2

    sources = sorted(dataset.rglob("*.md"), key=lambda path: path.as_posix().casefold())
    if not sources:
        print(f"error: no Markdown files found under {dataset}", file=sys.stderr)
        return 2

    changed = deleted = rejected = salvaged = 0
    errors: list[str] = []

    for source in sources:
        destination = source.with_suffix(".txt")
        try:
            result = clean_and_filter(extract_body(source))
        except ValueError as error:
            errors.append(f"{source}: {error}")
            continue

        relative = source.relative_to(dataset)
        if result.text is None:
            rejected += 1
            print(f"reject {relative}: {', '.join(result.reasons)}")
            if destination.exists():
                changed += 1
                if not args.check:
                    destination.unlink()
                    deleted += 1
            continue

        if result.salvaged:
            salvaged += 1
            print(f"salvage {relative}: removed sections with {', '.join(result.reasons)}")

        output = result.text.encode("utf-8")
        current = destination.read_bytes() if destination.exists() else None
        if current != output:
            changed += 1
            if not args.check:
                destination.write_bytes(output)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    if args.check and changed:
        print(
            f"{changed} outputs are missing, stale, or should be deleted "
            f"({rejected} rejected; {salvaged} partially salvaged)"
        )
        return 1

    if args.check:
        print(
            f"verified {len(sources) - rejected} text files "
            f"({rejected} rejected; {salvaged} partially salvaged)"
        )
    else:
        print(
            f"updated {changed - deleted} text files and deleted {deleted} rejected outputs "
            f"from {len(sources)} Markdown sources "
            f"({rejected} rejected; {salvaged} partially salvaged)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
