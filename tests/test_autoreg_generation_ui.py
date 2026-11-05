from __future__ import annotations

import unittest
from types import SimpleNamespace

from autoreg_generation_ui import GenerationApp
from generation import GenerationStep
from tokenizer import TokiPonaTokenizer


class _FakeText:
    def __init__(self, content: str = "") -> None:
        self.content = content

    def configure(self, **_kwargs: object) -> None:
        pass

    def index(self, index: str) -> str:
        if index != "end-1c":
            raise AssertionError(f"unexpected index: {index}")
        return f"1.{len(self.content)}"

    def insert(self, index: str, text: str, _tags: object = ()) -> None:
        if index != "end":
            raise AssertionError(f"unexpected index: {index}")
        self.content += text

    def delete(self, start: str, end: str) -> None:
        start_column = int(start.split(".", 1)[1])
        end_column = int(end.split(".", 1)[1])
        self.content = self.content[:start_column] + self.content[end_column:]

    def tag_configure(self, *_args: object, **_kwargs: object) -> None:
        pass

    def tag_bind(self, *_args: object, **_kwargs: object) -> None:
        pass

    def see(self, _index: str) -> None:
        pass


class GenerationUiTests(unittest.TestCase):
    def test_completed_spelling_displays_only_detokenized_word(self) -> None:
        tokenizer = TokiPonaTokenizer()
        app = GenerationApp.__new__(GenerationApp)
        app.generator = SimpleNamespace(tokenizer=tokenizer)
        app.generated_ids = []
        app.finished_with_eos = False
        app.pending_spell_marker = None
        app.tooltip = None
        app.output = _FakeText("toki")

        steps = (
            GenerationStep(tokenizer.spell_id, "<SPELL>", "", 0.1, ()),
            GenerationStep(tokenizer.letter_to_id["A"], "A", " A", 0.1, ()),
            GenerationStep(tokenizer.end_id, "<END>", "", 0.1, ()),
        )
        app._append_step(steps[0])
        self.assertIn("SPELL", app.output.content)
        app._append_step(steps[1])
        app._append_step(steps[2])

        self.assertEqual(app.output.content, "toki A")
        self.assertIsNone(app.pending_spell_marker)


if __name__ == "__main__":
    unittest.main()
