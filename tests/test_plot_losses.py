from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.plot_losses import MODEL_NAMES, load_losses


class PlotLossesTests(unittest.TestCase):
    def test_v2_dropout_model_has_a_fourth_tab(self) -> None:
        self.assertEqual(len(MODEL_NAMES), 4)
        self.assertEqual(MODEL_NAMES[-1], "medium-do0.1-v2")

    def test_load_losses_sorts_epochs_and_returns_both_curves(self) -> None:
        history = {
            "epochs": [
                {"epoch": 2, "training_loss": 2.5, "validation_loss": 2.75},
                {"epoch": 1, "training_loss": 3, "validation_loss": 3.25},
            ]
        }
        with patch.object(Path, "read_text", return_value=json.dumps(history)):
            epochs, training, validation = load_losses("history.json")

        self.assertEqual(epochs, [1, 2])
        self.assertEqual(training, [3.0, 2.5])
        self.assertEqual(validation, [3.25, 2.75])

    def test_load_losses_rejects_empty_history(self) -> None:
        with patch.object(Path, "read_text", return_value='{"epochs": []}'):
            with self.assertRaisesRegex(ValueError, "completed epochs"):
                load_losses("history.json")

    def test_load_losses_rejects_duplicate_epochs(self) -> None:
        history = {
            "epochs": [
                {"epoch": 1, "training_loss": 3.0, "validation_loss": 3.2},
                {"epoch": 1, "training_loss": 2.9, "validation_loss": 3.1},
            ]
        }
        with patch.object(Path, "read_text", return_value=json.dumps(history)):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_losses("history.json")


if __name__ == "__main__":
    unittest.main()
