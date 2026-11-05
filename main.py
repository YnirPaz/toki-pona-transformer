"""Command-line entry point for training a Toki Pona transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import MODEL_SIZES
from training import train


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    PROJECT_DIRECTORY / "data" / "processed" / "kulupu-lapo",
    PROJECT_DIRECTORY / "data" / "processed" / "tatoeba",
    PROJECT_DIRECTORY / "data" / "processed" / "wikipesija",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=MODEL_SIZES,
        default="small",
        help="model size to train or resume (default: small)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="training output directory (default: runs/<model>)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume from a checkpoint created in the output directory",
    )
    epoch_limit = parser.add_mutually_exclusive_group()
    epoch_limit.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="maximum total number of epochs (default: 50)",
    )
    epoch_limit.add_argument(
        "--resume-epochs",
        type=int,
        help="number of epochs to run from the resumed checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = args.output_dir or PROJECT_DIRECTORY / "runs" / args.model
    if args.epochs <= 0:
        raise SystemExit("error: --epochs must be positive")
    if args.resume_epochs is not None:
        if args.resume_epochs <= 0:
            raise SystemExit("error: --resume-epochs must be positive")
        if args.resume is None:
            raise SystemExit("error: --resume-epochs requires --resume")
    train(
        DEFAULT_DATASETS,
        output_directory,
        model_size=args.model,
        resume_checkpoint=args.resume,
        maximum_epochs=args.epochs,
        resume_epochs=args.resume_epochs,
    )


if __name__ == "__main__":
    # Start medium: python main.py --model medium
    # Resume medium for five epochs: python main.py --model medium --resume runs/medium/checkpoint.pt --resume-epochs 5
    # Start dropout medium: python main.py --model medium-do0.1
    # Resume dropout medium: python main.py --model medium-do0.1 --resume runs/medium-do0.1/checkpoint.pt
    # Start dropout v2 on all corpora: python main.py --model medium-do0.1-v2
    # Resume dropout v2 for 50 epochs: python main.py --model medium-do0.1-v2 --resume runs/medium-do0.1-v2/checkpoint.pt --resume-epochs 50

    main()
