from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from model import ModelConfig, TokiPonaTransformer, model_config_for_size
from tokenizer import TokiPonaTokenizer


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.tokenizer = TokiPonaTokenizer()
        self.config = ModelConfig(vocabulary_size=len(self.tokenizer))
        self.model = TokiPonaTransformer(self.config)

    def test_requested_architecture_and_parameter_count(self) -> None:
        self.assertEqual(self.config.layer_count, 4)
        self.assertEqual(self.config.embedding_dimension, 96)
        self.assertEqual(self.config.head_count, 4)
        self.assertEqual(self.config.head_dimension, 24)
        self.assertEqual(self.config.feed_forward_dimension, 384)
        self.assertEqual(self.config.context_size, 256)
        self.assertEqual(self.model.position_embedding.num_embeddings, 256)
        self.assertFalse(
            any(isinstance(module, torch.nn.Dropout) for module in self.model.modules())
        )
        self.assertEqual(self.model.parameter_count(), 493_824)

    def test_medium_architecture_and_parameter_count(self) -> None:
        config = model_config_for_size("medium", len(self.tokenizer))
        model = TokiPonaTransformer(config)

        self.assertEqual(config.layer_count, 4)
        self.assertEqual(config.embedding_dimension, 128)
        self.assertEqual(config.head_count, 4)
        self.assertEqual(config.head_dimension, 32)
        self.assertEqual(config.feed_forward_dimension, 512)
        self.assertEqual(config.context_size, 256)
        self.assertEqual(model.parameter_count(), 855_040)

    def test_dropout_medium_uses_dropout_at_all_requested_locations(self) -> None:
        config = model_config_for_size("medium-do0.1", len(self.tokenizer))
        model = TokiPonaTransformer(config)

        self.assertEqual(config.dropout_probability, 0.1)
        self.assertEqual(config.embedding_dimension, 128)
        self.assertEqual(config.feed_forward_dimension, 512)
        self.assertEqual(model.parameter_count(), 855_040)
        dropout_modules = [
            module for module in model.modules() if isinstance(module, torch.nn.Dropout)
        ]
        self.assertEqual(len(dropout_modules), 8)
        self.assertTrue(all(module.p == 0.1 for module in dropout_modules))

        attention = model.blocks[0].attention
        inputs = torch.randn(2, 8, config.embedding_dimension)
        with patch(
            "model.F.scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as attention_function:
            attention.train()
            attention(inputs)
            self.assertEqual(attention_function.call_args.kwargs["dropout_p"], 0.1)

            attention.eval()
            attention(inputs)
            self.assertEqual(attention_function.call_args.kwargs["dropout_p"], 0.0)

    def test_v2_dropout_medium_is_an_untrained_configuration_copy(self) -> None:
        original = model_config_for_size("medium-do0.1", len(self.tokenizer))
        version_two = model_config_for_size("medium-do0.1-v2", len(self.tokenizer))

        self.assertEqual(version_two, original)
        self.assertEqual(version_two.dropout_probability, 0.1)

    def test_embedding_and_unembedding_weights_are_tied(self) -> None:
        self.assertIs(self.model.token_embedding.weight, self.model.unembedding.weight)

    def test_forward_shape_and_loss(self) -> None:
        inputs = torch.randint(0, len(self.tokenizer), (2, 16))
        targets = torch.randint(0, len(self.tokenizer), (2, 16))
        logits, loss = self.model(inputs, targets)
        self.assertEqual(logits.shape, (2, 16, len(self.tokenizer)))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))

    def test_attention_is_causal(self) -> None:
        self.model.eval()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 9, 10]])
        with torch.no_grad():
            first_logits, _ = self.model(first)
            second_logits, _ = self.model(second)
        torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])

    def test_context_limit_is_enforced(self) -> None:
        too_long = torch.zeros((1, 257), dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "exceeds context size"):
            self.model(too_long)

    def test_cached_inference_matches_full_forward(self) -> None:
        self.model.eval()
        tokens = torch.randint(0, len(self.tokenizer), (2, 9))
        with torch.no_grad():
            full_logits, _ = self.model(tokens)
            first_logits, cache = self.model.forward_with_cache(tokens[:, :6])
            next_logits, cache = self.model.forward_with_cache(tokens[:, 6:7], cache)
            final_logits, cache = self.model.forward_with_cache(tokens[:, 7:], cache)

        torch.testing.assert_close(first_logits, full_logits[:, :6], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(next_logits, full_logits[:, 6:7], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(final_logits, full_logits[:, 7:], rtol=1e-5, atol=1e-5)
        self.assertTrue(all(key.shape[-2] == 9 for key, _ in cache))


if __name__ == "__main__":
    unittest.main()
