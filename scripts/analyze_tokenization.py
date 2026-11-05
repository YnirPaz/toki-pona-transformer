#!/usr/bin/env python3
"""Report corpus statistics for the proposed Toki Pona tokenization scheme.

This script only analyzes ``.txt`` files; it never writes tokenized data.

Rules modeled here:

* Each of the 120 nimi pu and 17 additional nimi ku suli has a word token.
* ``ali`` is accepted as the alternate spelling of ``ale`` and uses the same
  word token, so there are still 137 word-token IDs.
* Known words are recognized case-insensitively.  Thus ``li``, ``Li``, and
  ``LI`` all use the same word token.
* Every unfamiliar run of Unicode letters is surrounded by ``<SPELL>`` and
  ``<END>`` and represented by one token per Unicode letter/grapheme cluster.
* Ordinary horizontal spaces separate tokens and are not themselves tokens.
  Each newline, punctuation mark, digit, symbol, or other formatting character
  is one mark token.  CRLF is treated as a single newline token.
* One ``<EOS>`` token is appended to every file/sample.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# The 120 headwords taught in Toki Pona: The Language of Good (pu).
NIMI_PU = frozenset(
    """
    a akesi ala alasa ale anpa ante anu awen e en esun ijo ike ilo insa jaki
    jan jelo jo kala kalama kama kasi ken kepeken kili kiwen ko kon kule kulupu
    kute la lape laso lawa len lete li lili linja lipu loje lon luka lukin lupa
    ma mama mani meli mi mije moku moli monsi mu mun musi mute nanpa nasa nasin
    nena ni nimi noka o olin ona open pakala pali palisa pan pana pi pilin pimeja
    pini pipi poka poki pona pu sama seli selo seme sewi sijelo sike sin sina
    sinpin sitelen sona soweli suli suno supa suwi tan taso tawa telo tenpo toki
    tomo tu unpa uta utala walo wan waso wawa weka wile
    """.split()
)

# The 16-word list labeled "nimi ku suli" in ku, plus "ku" itself.
NIMI_KU_SULI_ADDITIONS = frozenset(
    """
    epiku jasima kijetesantakalu kin kipisi kokosila ku lanpan leko meso
    misikeke monsuta n namako oko soko tonsi
    """.split()
)

WORD_TOKENS = NIMI_PU | NIMI_KU_SULI_ADDITIONS
WORD_FORMS = {word: word for word in WORD_TOKENS} | {"ali": "ale"}

assert len(NIMI_PU) == 120
assert len(NIMI_KU_SULI_ADDITIONS) == 17
assert len(WORD_TOKENS) == 137


@dataclass
class TokenCounts:
    words: int = 0
    letters: int = 0
    spell: int = 0
    end: int = 0
    marks: int = 0
    eos: int = 0
    observed_words: set[str] = field(default_factory=set)
    observed_letters: set[str] = field(default_factory=set)
    observed_marks: set[str] = field(default_factory=set)
    unfamiliar_words: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return self.words + self.letters + self.spell + self.end + self.marks + self.eos

    def add(self, other: "TokenCounts") -> None:
        self.words += other.words
        self.letters += other.letters
        self.spell += other.spell
        self.end += other.end
        self.marks += other.marks
        self.eos += other.eos
        self.observed_words.update(other.observed_words)
        self.observed_letters.update(other.observed_letters)
        self.observed_marks.update(other.observed_marks)
        self.unfamiliar_words.update(other.unfamiliar_words)


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_combining_mark(character: str) -> bool:
    return unicodedata.category(character).startswith("M")


def _read_letter_cluster(text: str, index: int) -> tuple[str, int]:
    """Read one base letter and any following Unicode combining marks."""
    end = index + 1
    while end < len(text) and _is_combining_mark(text[end]):
        end += 1
    return text[index:end], end


def analyze_text(text: str) -> TokenCounts:
    counts = TokenCounts(eos=1)
    text = unicodedata.normalize("NFC", text)
    index = 0

    while index < len(text):
        character = text[index]

        if character == "\r":
            if index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
            counts.marks += 1
            counts.observed_marks.add("\n")
            index += 1
            continue

        if character == "\n":
            counts.marks += 1
            counts.observed_marks.add("\n")
            index += 1
            continue

        if _is_letter(character):
            clusters: list[str] = []
            while index < len(text) and _is_letter(text[index]):
                cluster, index = _read_letter_cluster(text, index)
                clusters.append(cluster)

            word = "".join(clusters)
            word_token = WORD_FORMS.get(word.casefold())
            if word_token is not None:
                counts.words += 1
                counts.observed_words.add(word_token)
            else:
                counts.spell += 1
                counts.letters += len(clusters)
                counts.end += 1
                counts.observed_letters.update(clusters)
                counts.unfamiliar_words[word] += 1
            continue

        if character.isspace() and character not in "\t\v\f":
            # Spaces, including non-breaking Unicode spaces, are separators.
            index += 1
            continue

        counts.marks += 1
        counts.observed_marks.add(character)
        index += 1

    return counts


def _percent(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _result_dict(
    dataset: Path,
    sources: list[Path],
    totals: TokenCounts,
    sample_lengths: list[int],
) -> dict[str, object]:
    spelling_total = totals.letters + totals.spell + totals.end
    defined_vocabulary_size = (
        len(WORD_TOKENS)
        + len(totals.observed_letters)
        + len(totals.observed_marks)
        + 3  # <SPELL>, <END>, and <EOS>
    )

    return {
        "dataset": str(dataset),
        "samples": len(sources),
        "defined_word_token_ids": len(WORD_TOKENS),
        "recognized_word_spellings": len(WORD_FORMS),
        "observed_letter_token_ids": len(totals.observed_letters),
        "observed_mark_token_ids": len(totals.observed_marks),
        "tokenizer_vocabulary_size_on_this_corpus": defined_vocabulary_size,
        "total_token_instances": totals.total,
        "mean_tokens_per_sample": statistics.fmean(sample_lengths),
        "median_tokens_per_sample": statistics.median(sample_lengths),
        "counts": {
            "word": totals.words,
            "letter": totals.letters,
            "spell": totals.spell,
            "end": totals.end,
            "spelling_total": spelling_total,
            "mark": totals.marks,
            "eos": totals.eos,
        },
        "percentages": {
            "word": _percent(totals.words, totals.total),
            "letter": _percent(totals.letters, totals.total),
            "spell": _percent(totals.spell, totals.total),
            "end": _percent(totals.end, totals.total),
            "spelling_total": _percent(spelling_total, totals.total),
            "mark": _percent(totals.marks, totals.total),
            "eos": _percent(totals.eos, totals.total),
        },
    }


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
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        print(f"error: dataset directory does not exist: {dataset}", file=sys.stderr)
        return 2

    sources = sorted(dataset.rglob("*.txt"), key=lambda path: path.as_posix().casefold())
    if not sources:
        print(f"error: no text files found under {dataset}", file=sys.stderr)
        return 2

    totals = TokenCounts()
    sample_lengths: list[int] = []
    for source in sources:
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            print(f"error: {source}: not valid UTF-8: {error}", file=sys.stderr)
            return 1
        sample = analyze_text(text)
        totals.add(sample)
        sample_lengths.append(sample.total)

    result = _result_dict(dataset, sources, totals, sample_lengths)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    counts = result["counts"]
    percentages = result["percentages"]
    assert isinstance(counts, dict)
    assert isinstance(percentages, dict)

    print(f"Dataset: {dataset}")
    print(f"Samples (.txt files): {result['samples']:,}")
    print(f"Defined word-token IDs: {result['defined_word_token_ids']:,}")
    print(
        "Observed tokenizer vocabulary: "
        f"{result['tokenizer_vocabulary_size_on_this_corpus']:,} token IDs "
        f"({result['observed_letter_token_ids']:,} letters, "
        f"{result['observed_mark_token_ids']:,} marks)"
    )
    print(f"Total token instances: {result['total_token_instances']:,}")
    print(f"Mean tokens per sample: {result['mean_tokens_per_sample']:,.2f}")
    print(f"Median tokens per sample: {result['median_tokens_per_sample']:,.1f}")
    print("Token-instance breakdown:")
    print(f"  known words       {counts['word']:>12,}  {percentages['word']:>7.3f}%")
    print(f"  spelled letters   {counts['letter']:>12,}  {percentages['letter']:>7.3f}%")
    print(f"  <SPELL>           {counts['spell']:>12,}  {percentages['spell']:>7.3f}%")
    print(f"  <END>             {counts['end']:>12,}  {percentages['end']:>7.3f}%")
    print(
        f"  spelling subtotal {counts['spelling_total']:>12,}  "
        f"{percentages['spelling_total']:>7.3f}%"
    )
    print(f"  marks             {counts['mark']:>12,}  {percentages['mark']:>7.3f}%")
    print(f"  <EOS>             {counts['eos']:>12,}  {percentages['eos']:>7.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
