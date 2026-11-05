"""Tokenizer for the Toki Pona language model.

Known pu and ku suli words use one case-insensitive token each. Unfamiliar
words are represented as case-sensitive ASCII letters between ``<SPELL>`` and
``<END>``. Spaces are discarded, while newlines and a deliberately small set
of useful punctuation marks are retained.
"""

from __future__ import annotations

import json
import string
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence


TOKENIZER_FORMAT = "toki-pona-transformer-tokenizer"
TOKENIZER_VERSION = 1

SPECIAL_TOKENS = ("<UNK>", "<BOS>", "<EOS>", "<SPELL>", "<END>")

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

# The 16-word nimi ku suli list, plus "ku" itself.
NIMI_KU_SULI_ADDITIONS = frozenset(
    """
    epiku jasima kijetesantakalu kin kipisi kokosila ku lanpan leko meso
    misikeke monsuta n namako oko soko tonsi
    """.split()
)

WORD_TOKENS = NIMI_PU | NIMI_KU_SULI_ADDITIONS
WORD_FORMS = {word: word for word in WORD_TOKENS} | {"ali": "ale"}
LETTERS = tuple(string.ascii_lowercase + string.ascii_uppercase)

# Order is part of the checkpoint format and must remain stable.
MARKS = (
    "\n",
    ".",
    ",",
    ":",
    "!",
    "?",
    ";",
    "-",
    '"',
    "'",
    "(",
    ")",
    *tuple(string.digits),
    "“",
    "”",
    "‘",
    "’",
    "…",
    "—",
    "–",
    "·",
    "+",
    "/",
    "=",
    "#",
    "‹",
    "›",
    "«",
    "»",
    "｢",
    "｣",
    "→",
)

assert len(NIMI_PU) == 120
assert len(NIMI_KU_SULI_ADDITIONS) == 17
assert len(WORD_TOKENS) == 137
assert len(MARKS) == 41


def _word_token(word: str) -> str:
    return f"<WORD:{word}>"


def _letter_token(letter: str) -> str:
    return f"<LETTER:{letter}>"


def _mark_token(mark: str) -> str:
    label = "NEWLINE" if mark == "\n" else mark
    return f"<MARK:{label}>"


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_extension(character: str) -> bool:
    codepoint = ord(character)
    return (
        unicodedata.category(character).startswith("M")
        or codepoint in (0xFE0E, 0xFE0F)
        or 0x1F3FB <= codepoint <= 0x1F3FF
    )


def _read_letter_cluster(text: str, index: int) -> tuple[str, int]:
    end = index + 1
    while end < len(text) and _is_extension(text[end]):
        end += 1
    return text[index:end], end


def _read_unknown_unit(text: str, index: int) -> int:
    """Return the end of one approximate grapheme, including emoji joins."""
    end = index + 1
    while end < len(text):
        while end < len(text) and _is_extension(text[end]):
            end += 1
        if end + 1 < len(text) and text[end] == "\u200d":
            end += 2
            continue
        break
    return end


class TokiPonaTokenizer:
    """Fixed-vocabulary tokenizer used by the language model."""

    def __init__(self) -> None:
        vocabulary = [
            *SPECIAL_TOKENS,
            *(_word_token(word) for word in sorted(WORD_TOKENS)),
            *(_letter_token(letter) for letter in LETTERS),
            *(_mark_token(mark) for mark in MARKS),
        ]
        self.vocabulary = tuple(vocabulary)
        self.token_to_id = {token: index for index, token in enumerate(vocabulary)}

        self.unk_id = self.token_to_id["<UNK>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.spell_id = self.token_to_id["<SPELL>"]
        self.end_id = self.token_to_id["<END>"]

        self.word_to_id = {
            form: self.token_to_id[_word_token(canonical)]
            for form, canonical in WORD_FORMS.items()
        }
        self.id_to_word = {
            self.token_to_id[_word_token(word)]: word for word in WORD_TOKENS
        }
        self.letter_to_id = {
            letter: self.token_to_id[_letter_token(letter)] for letter in LETTERS
        }
        self.id_to_letter = {token_id: letter for letter, token_id in self.letter_to_id.items()}
        self.mark_to_id = {mark: self.token_to_id[_mark_token(mark)] for mark in MARKS}
        self.id_to_mark = {token_id: mark for mark, token_id in self.mark_to_id.items()}

    def __len__(self) -> int:
        return len(self.vocabulary)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode one sequence, inserting BOS and EOS by default."""
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        token_ids: list[int] = [self.bos_id] if add_bos else []
        index = 0

        while index < len(text):
            character = text[index]

            if character.isspace() and character != "\n":
                index += 1
                continue

            if _is_letter(character):
                clusters: list[str] = []
                while index < len(text) and _is_letter(text[index]):
                    cluster, index = _read_letter_cluster(text, index)
                    clusters.append(cluster)

                word = "".join(clusters)
                known_id = self.word_to_id.get(word.casefold())
                if known_id is not None:
                    token_ids.append(known_id)
                else:
                    token_ids.append(self.spell_id)
                    token_ids.extend(
                        self.letter_to_id.get(cluster, self.unk_id)
                        for cluster in clusters
                    )
                    token_ids.append(self.end_id)
                continue

            mark_id = self.mark_to_id.get(character)
            if mark_id is not None:
                token_ids.append(mark_id)
                index += 1
                continue

            index = _read_unknown_unit(text, index)
            token_ids.append(self.unk_id)

        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode IDs with canonical spaces, since spaces are not tokenized."""
        units: list[tuple[str, str]] = []
        spelling: list[str] | None = None

        def finish_spelling() -> None:
            nonlocal spelling
            if spelling is not None:
                units.append(("word", "".join(spelling)))
                spelling = None

        for token_id in token_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < len(self):
                raise ValueError(f"invalid token ID: {token_id!r}")

            if token_id in (self.bos_id, self.eos_id):
                finish_spelling()
                if not skip_special_tokens:
                    units.append(("word", self.vocabulary[token_id]))
            elif token_id == self.spell_id:
                finish_spelling()
                spelling = []
            elif token_id == self.end_id:
                finish_spelling()
            elif token_id == self.unk_id:
                if spelling is not None:
                    spelling.append("�")
                else:
                    units.append(("word", "�"))
            elif token_id in self.id_to_word:
                finish_spelling()
                units.append(("word", self.id_to_word[token_id]))
            elif token_id in self.id_to_letter:
                letter = self.id_to_letter[token_id]
                if spelling is not None:
                    spelling.append(letter)
                else:
                    units.append(("word", letter))
            else:
                finish_spelling()
                units.append(("mark", self.id_to_mark[token_id]))

        finish_spelling()
        return _render_with_canonical_spaces(units)

    def save(self, path: str | Path) -> None:
        """Save the complete, ordered tokenizer definition as JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": TOKENIZER_FORMAT,
            "version": TOKENIZER_VERSION,
            "normalization": "NFC",
            "spaces": "discarded",
            "vocabulary": list(self.vocabulary),
            "word_forms": dict(sorted(WORD_FORMS.items())),
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TokiPonaTokenizer":
        """Load a checkpoint and reject incompatible tokenizer definitions."""
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        tokenizer = cls()
        if not isinstance(payload, dict):
            raise ValueError("tokenizer checkpoint must contain a JSON object")
        if payload.get("format") != TOKENIZER_FORMAT:
            raise ValueError("unrecognized tokenizer checkpoint format")
        if payload.get("version") != TOKENIZER_VERSION:
            raise ValueError(f"unsupported tokenizer version: {payload.get('version')!r}")
        if payload.get("vocabulary") != list(tokenizer.vocabulary):
            raise ValueError("checkpoint vocabulary does not match this tokenizer version")
        if payload.get("word_forms") != dict(sorted(WORD_FORMS.items())):
            raise ValueError("checkpoint word forms do not match this tokenizer version")
        return tokenizer


_OPENING_MARKS = {"(", "“", "‘", "‹", "«", "｢", "#"}
_CLOSING_MARKS = {")", ".", ",", ":", "!", "?", ";", "”", "’", "…", "·", "›", "»", "｣"}
_SPACED_MARKS = {"—", "–", "+", "=", "→"}


def _render_with_canonical_spaces(units: Sequence[tuple[str, str]]) -> str:
    output = ""
    previous = "start"
    double_quote_open = False
    single_quote_open = False

    for kind, value in units:
        if kind == "word":
            if output and not output.endswith((" ", "\n")) and previous in {
                "word",
                "closing",
                "number",
            }:
                output += " "
            output += value
            previous = "word"
            continue

        if value == "\n":
            output = output.rstrip(" ") + "\n"
            previous = "newline"
        elif value.isdigit():
            if output and not output.endswith((" ", "\n")) and previous in {"word", "closing"}:
                output += " "
            output += value
            previous = "number"
        elif value == '"':
            if double_quote_open:
                output += value
                previous = "closing"
            else:
                if output and not output.endswith((" ", "\n")) and previous in {"word", "closing", "number"}:
                    output += " "
                output += value
                previous = "opening"
            double_quote_open = not double_quote_open
        elif value == "'":
            if single_quote_open:
                output += value
                previous = "closing"
            else:
                if output and not output.endswith((" ", "\n")) and previous in {"word", "closing", "number"}:
                    output += " "
                output += value
                previous = "opening"
            single_quote_open = not single_quote_open
        elif value in _OPENING_MARKS:
            if output and not output.endswith((" ", "\n")) and previous in {"word", "closing", "number"}:
                output += " "
            output += value
            previous = "opening"
        elif value in _CLOSING_MARKS:
            output = output.rstrip(" ") + value
            previous = "closing"
        elif value in _SPACED_MARKS:
            if output and not output.endswith((" ", "\n")):
                output += " "
            output += value + " "
            previous = "separator"
        else:
            output += value
            previous = "attached"

    return output.rstrip(" ")

