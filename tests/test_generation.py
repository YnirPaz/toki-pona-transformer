from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from generation import AutoregressiveGenerator
from model import ModelConfig, TokiPonaTransformer
from tokenizer import TokiPonaTokenizer


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.tokenizer = TokiPonaTokenizer()
        config = ModelConfig(
            vocabulary_size=len(self.tokenizer),
            context_size=8,
            embedding_dimension=16,
            layer_count=1,
            head_count=2,
            feed_forward_dimension=32,
        )
        self.generator = AutoregressiveGenerator(
            TokiPonaTransformer(config),
            self.tokenizer,
        )

    def test_empty_prompt_starts_with_bos(self) -> None:
        self.assertEqual(self.generator.encode_prompt(""), [self.tokenizer.bos_id])

    def test_generation_has_probabilities_and_honors_limit(self) -> None:
        steps = list(
            self.generator.generate(
                self.generator.encode_prompt("toki"),
                temperature=1.0,
                max_new_tokens=3,
                top_count=4,
            )
        )
        self.assertLessEqual(len(steps), 3)
        self.assertGreater(len(steps), 0)
        self.assertTrue(all(len(step.top_probabilities) == 4 for step in steps))
        self.assertTrue(all(0.0 <= step.probability <= 1.0 for step in steps))

    def test_generation_slides_past_context_limit(self) -> None:
        prompt = [self.tokenizer.bos_id] * 8
        selected_id = self.tokenizer.word_to_id["toki"]
        with patch.object(
            self.generator,
            "_sample",
            return_value=(selected_id, 1.0, ()),
        ):
            steps = list(self.generator.generate(prompt, max_new_tokens=2))
        self.assertEqual(len(steps), 2)

    def test_rejects_nonpositive_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature"):
            list(self.generator.generate([self.tokenizer.bos_id], temperature=0.0))


if __name__ == "__main__":
    unittest.main()
