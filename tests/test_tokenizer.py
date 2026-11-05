from __future__ import annotations

import json
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from streams import (
    build_evaluation_stream,
    build_training_stream,
    build_validation_stream,
    iter_causal_windows,
)
from tokenizer import TokiPonaTokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = TokiPonaTokenizer()

    def names(self, text: str) -> list[str]:
        return [self.tokenizer.vocabulary[token_id] for token_id in self.tokenizer.encode(text)]

    def test_vocabulary_is_fixed_and_has_no_padding(self) -> None:
        self.assertEqual(len(self.tokenizer), 235)
        self.assertNotIn("<PAD>", self.tokenizer.token_to_id)

    def test_known_words_ignore_case_and_ali_aliases_ale(self) -> None:
        ids = self.tokenizer.encode("TOKI Pona ali")
        self.assertEqual(
            ids,
            [
                self.tokenizer.bos_id,
                self.tokenizer.word_to_id["toki"],
                self.tokenizer.word_to_id["pona"],
                self.tokenizer.word_to_id["ale"],
                self.tokenizer.eos_id,
            ],
        )
        self.assertEqual(self.tokenizer.word_to_id["ali"], self.tokenizer.word_to_id["ale"])

    def test_unknown_words_preserve_ascii_letter_case(self) -> None:
        self.assertEqual(
            self.names("jan Alice"),
            [
                "<BOS>",
                "<WORD:jan>",
                "<SPELL>",
                "<LETTER:A>",
                "<LETTER:l>",
                "<LETTER:i>",
                "<LETTER:c>",
                "<LETTER:e>",
                "<END>",
                "<EOS>",
            ],
        )

    def test_spaces_disappear_but_newlines_and_punctuation_remain(self) -> None:
        self.assertEqual(
            self.names("toki,  pona\r\n\r\nale · 12"),
            [
                "<BOS>",
                "<WORD:toki>",
                "<MARK:,>",
                "<WORD:pona>",
                "<MARK:NEWLINE>",
                "<MARK:NEWLINE>",
                "<WORD:ale>",
                "<MARK:·>",
                "<MARK:1>",
                "<MARK:2>",
                "<EOS>",
            ],
        )

    def test_unsupported_letters_and_emoji_become_unknown(self) -> None:
        self.assertEqual(
            self.names("AΩ 👩‍💻"),
            [
                "<BOS>",
                "<SPELL>",
                "<LETTER:A>",
                "<UNK>",
                "<END>",
                "<UNK>",
                "<EOS>",
            ],
        )

    def test_decode_inserts_canonical_spaces(self) -> None:
        ids = self.tokenizer.encode('toki, pona! "jan Alice"\n\nale')
        self.assertEqual(self.tokenizer.decode(ids), 'toki, pona! "jan Alice"\n\nale')

    def test_checkpoint_round_trip_and_validation(self) -> None:
        saved: dict[str, str] = {}

        def remember_write(data: str, **_: object) -> int:
            saved["data"] = data
            return len(data)

        path = Path("tokenizer-test.json")
        with patch.object(Path, "write_text", side_effect=remember_write):
            self.tokenizer.save(path)

        with patch.object(Path, "read_text", return_value=saved["data"]):
            loaded = TokiPonaTokenizer.load(path)
        self.assertEqual(loaded.vocabulary, self.tokenizer.vocabulary)

        payload = json.loads(saved["data"])
        payload["vocabulary"][0] = "<BROKEN>"
        with patch.object(Path, "read_text", return_value=json.dumps(payload)):
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                TokiPonaTokenizer.load(path)


class StreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Path(__file__).parent / "fixtures" / "dataset"
        self.second_dataset = Path(__file__).parent / "fixtures" / "dataset_secondary"
        self.tokenizer = TokiPonaTokenizer()

    def test_fixed_streams_repeat_exactly(self) -> None:
        self.assertEqual(
            build_validation_stream(self.dataset, self.tokenizer),
            build_validation_stream(self.dataset, self.tokenizer),
        )
        self.assertEqual(
            build_evaluation_stream(self.dataset, self.tokenizer),
            build_evaluation_stream(self.dataset, self.tokenizer),
        )

    def test_training_stream_accepts_reproducible_rng(self) -> None:
        first = build_training_stream(self.dataset, self.tokenizer, rng=random.Random(1))
        repeated = build_training_stream(self.dataset, self.tokenizer, rng=random.Random(1))
        different = build_training_stream(self.dataset, self.tokenizer, rng=random.Random(2))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertEqual(first.count(self.tokenizer.bos_id), 5)
        self.assertEqual(first.count(self.tokenizer.eos_id), 5)

    def test_multiple_datasets_are_mixed_as_one_document_pool(self) -> None:
        first = build_training_stream(
            (self.dataset, self.second_dataset),
            self.tokenizer,
            rng=random.Random(3),
        )
        repeated = build_training_stream(
            (self.dataset, self.second_dataset),
            self.tokenizer,
            rng=random.Random(3),
        )

        self.assertEqual(first, repeated)
        self.assertEqual(first.count(self.tokenizer.bos_id), 6)
        self.assertEqual(first.count(self.tokenizer.eos_id), 6)

    def test_causal_windows_shift_targets_and_drop_remainder(self) -> None:
        windows = list(iter_causal_windows(list(range(12)), context_size=4))
        self.assertEqual(windows, [([0, 1, 2, 3], [1, 2, 3, 4]), ([4, 5, 6, 7], [5, 6, 7, 8])])


if __name__ == "__main__":
    unittest.main()

