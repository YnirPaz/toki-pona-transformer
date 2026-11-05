from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_models import select_checkpoint


class EvaluateModelsTests(unittest.TestCase):
    def test_selects_highest_numbered_best_checkpoint(self) -> None:
        checkpoints = [
            Path("run/best/epoch_999.pt"),
            Path("run/best/epoch_1000.pt"),
        ]
        with patch.object(Path, "glob", return_value=checkpoints):
            selected = select_checkpoint("run")

        self.assertEqual(selected.name, "epoch_1000.pt")

    def test_selects_latest_checkpoint(self) -> None:
        checkpoint = Path("run/checkpoint.pt")
        with patch.object(Path, "is_file", return_value=True):
            selected = select_checkpoint("run", "latest")

        self.assertEqual(selected, checkpoint)

    def test_rejects_unknown_checkpoint_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "best.*latest"):
            select_checkpoint("unused", "oldest")


if __name__ == "__main__":
    unittest.main()
