"""Tests for the training command-line entry point."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import main


class MainTests(unittest.TestCase):
    def test_default_datasets_are_the_three_prepared_corpora(self) -> None:
        self.assertEqual(
            [path.name for path in main.DEFAULT_DATASETS],
            ["kulupu-lapo", "tatoeba", "wikipesija"],
        )
        self.assertTrue(all(path.parent.name == "processed" for path in main.DEFAULT_DATASETS))

    @patch("main.train")
    def test_resume_epochs_is_forwarded(self, train_mock) -> None:
        with patch(
            "sys.argv",
            ["main.py", "--resume", "checkpoint.pt", "--resume-epochs", "5"],
        ):
            main.main()

        train_mock.assert_called_once()
        self.assertEqual(train_mock.call_args.kwargs["resume_checkpoint"], Path("checkpoint.pt"))
        self.assertEqual(train_mock.call_args.kwargs["resume_epochs"], 5)
        self.assertEqual(train_mock.call_args.kwargs["model_size"], "small")
        self.assertEqual(train_mock.call_args.args[0], main.DEFAULT_DATASETS)

    @patch("main.train")
    def test_medium_model_uses_its_own_default_output_directory(self, train_mock) -> None:
        with patch("sys.argv", ["main.py", "--model", "medium"]):
            main.main()

        train_mock.assert_called_once()
        self.assertEqual(train_mock.call_args.kwargs["model_size"], "medium")
        output_directory = train_mock.call_args.args[1]
        self.assertEqual(output_directory.name, "medium")
        self.assertEqual(output_directory.parent.name, "runs")

    @patch("main.train")
    def test_v2_dropout_model_uses_fresh_output_directory(self, train_mock) -> None:
        with patch("sys.argv", ["main.py", "--model", "medium-do0.1-v2"]):
            main.main()

        train_mock.assert_called_once()
        self.assertEqual(train_mock.call_args.kwargs["model_size"], "medium-do0.1-v2")
        self.assertEqual(train_mock.call_args.args[1].name, "medium-do0.1-v2")

    @patch("main.train")
    def test_resume_epochs_requires_resume_checkpoint(self, train_mock) -> None:
        with patch("sys.argv", ["main.py", "--resume-epochs", "5"]):
            with self.assertRaisesRegex(SystemExit, "requires --resume"):
                main.main()

        train_mock.assert_not_called()

    @patch("main.train")
    def test_resume_epochs_must_be_positive(self, train_mock) -> None:
        with patch(
            "sys.argv",
            ["main.py", "--resume", "checkpoint.pt", "--resume-epochs", "0"],
        ):
            with self.assertRaisesRegex(SystemExit, "must be positive"):
                main.main()

        train_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
