"""Display model loss histories in switchable tabs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("small", "medium", "medium-do0.1", "medium-do0.1-v2")
MODEL_LABELS = {
    "small": "Small",
    "medium": "Medium",
    "medium-do0.1": "Medium do0.1",
    "medium-do0.1-v2": "Medium do0.1 v2",
}
DEFAULT_HISTORY_PATHS = {
    model_name: PROJECT_DIRECTORY / "runs" / model_name / "history.json"
    for model_name in MODEL_NAMES
}
LossData = tuple[list[int], list[float], list[float]]


def load_losses(history_path: str | Path) -> LossData:
    """Load and validate epoch loss values from ``history.json``."""
    path = Path(history_path)
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("epochs"), list):
        raise ValueError("history must contain an 'epochs' list")
    if not payload["epochs"]:
        raise ValueError("history does not contain any completed epochs")

    rows: list[tuple[int, float, float]] = []
    for index, entry in enumerate(payload["epochs"], start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"epoch entry {index} is not an object")

        epoch = entry.get("epoch")
        training_loss = entry.get("training_loss")
        validation_loss = entry.get("validation_loss")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError(f"epoch entry {index} has an invalid epoch number")
        if (
            isinstance(training_loss, bool)
            or not isinstance(training_loss, (int, float))
            or not math.isfinite(training_loss)
        ):
            raise ValueError(f"epoch {epoch} has an invalid training loss")
        if (
            isinstance(validation_loss, bool)
            or not isinstance(validation_loss, (int, float))
            or not math.isfinite(validation_loss)
        ):
            raise ValueError(f"epoch {epoch} has an invalid validation loss")
        rows.append((epoch, float(training_loss), float(validation_loss)))

    rows.sort(key=lambda row: row[0])
    epochs = [row[0] for row in rows]
    if len(epochs) != len(set(epochs)):
        raise ValueError("history contains duplicate epoch numbers")

    return (
        epochs,
        [row[1] for row in rows],
        [row[2] for row in rows],
    )


def _draw_loss_plot(axes, model_name: str, losses: LossData | None) -> None:
    from matplotlib.ticker import MaxNLocator

    axes.clear()
    axes.set_title(f"{MODEL_LABELS[model_name]} Model Loss by Epoch")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Cross-entropy loss")
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.grid(True, alpha=0.25)

    if losses is None:
        axes.set_ylim(0, 1)
        axes.text(
            0.5,
            0.5,
            "No completed epochs yet",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes.transAxes,
            fontsize=13,
            color="#666666",
        )
        return

    epochs, training_losses, validation_losses = losses
    axes.plot(
        epochs,
        training_losses,
        color="#1f77b4",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Training loss",
    )
    axes.plot(
        epochs,
        validation_losses,
        color="#ff7f0e",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Validation loss",
    )

    best_index = min(range(len(validation_losses)), key=validation_losses.__getitem__)
    axes.scatter(
        epochs[best_index],
        validation_losses[best_index],
        color="#2ca02c",
        marker="*",
        s=130,
        zorder=3,
        label=f"Best validation (epoch {epochs[best_index]})",
    )
    axes.set_ylim(bottom=0)
    axes.legend()


def plot_loss_tabs(
    histories: dict[str, LossData | None],
) -> tuple[Any, Callable[[str], None], str]:
    """Create one figure with tabs that show one model's losses at a time."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required; install it with 'python -m pip install matplotlib'."
        ) from error

    figure, axes = plt.subplots(figsize=(9, 5.5))
    figure.subplots_adjust(left=0.1, right=0.96, bottom=0.11, top=0.82)

    buttons = {}
    button_width = 0.18
    button_gap = 0.015
    group_width = (
        len(MODEL_NAMES) * button_width + (len(MODEL_NAMES) - 1) * button_gap
    )
    group_left = (1.0 - group_width) / 2.0
    for index, model_name in enumerate(MODEL_NAMES):
        button_axes = figure.add_axes(
            [group_left + index * (button_width + button_gap), 0.88, button_width, 0.07]
        )
        buttons[model_name] = Button(
            button_axes,
            MODEL_LABELS[model_name],
            color="#eeeeee",
            hovercolor="#dce9f7",
        )

    def select_tab(model_name: str) -> None:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"unknown loss tab: {model_name}")
        _draw_loss_plot(axes, model_name, histories.get(model_name))
        for name, button in buttons.items():
            button.ax.set_facecolor("#b9d7f5" if name == model_name else "#eeeeee")
        figure.canvas.draw_idle()

    for model_name, button in buttons.items():
        button.on_clicked(lambda _event, name=model_name: select_tab(name))

    initial_model = next(
        (name for name in MODEL_NAMES if histories.get(name) is not None),
        "small",
    )
    select_tab(initial_model)

    # Retain the widgets for the lifetime of the figure.
    setattr(figure, "_loss_tab_buttons", buttons)
    return figure, select_tab, initial_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small-history",
        type=Path,
        default=DEFAULT_HISTORY_PATHS["small"],
        help="small-model history (default: runs/small/history.json)",
    )
    parser.add_argument(
        "--medium-history",
        type=Path,
        default=DEFAULT_HISTORY_PATHS["medium"],
        help="medium-model history (default: runs/medium/history.json)",
    )
    parser.add_argument(
        "--medium-dropout-history",
        type=Path,
        default=DEFAULT_HISTORY_PATHS["medium-do0.1"],
        help="dropout-medium history (default: runs/medium-do0.1/history.json)",
    )
    parser.add_argument(
        "--medium-dropout-v2-history",
        type=Path,
        default=DEFAULT_HISTORY_PATHS["medium-do0.1-v2"],
        help="dropout-medium-v2 history (default: runs/medium-do0.1-v2/history.json)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        metavar="PATH",
        help="save each available tab using PATH with a model-name suffix",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open a window (useful together with --save)",
    )
    return parser.parse_args()


def _model_output_path(base_path: Path, model_name: str) -> Path:
    suffix = base_path.suffix or ".png"
    stem = base_path.stem if base_path.suffix else base_path.name
    return base_path.with_name(f"{stem}_{model_name}{suffix}")


def main() -> int:
    args = parse_args()
    if args.no_show and args.save is None:
        print("error: --no-show has no effect unless --save is also provided", file=sys.stderr)
        return 2

    history_paths = {
        "small": args.small_history,
        "medium": args.medium_history,
        "medium-do0.1": args.medium_dropout_history,
        "medium-do0.1-v2": args.medium_dropout_v2_history,
    }
    histories: dict[str, LossData | None] = {}
    for model_name, path in history_paths.items():
        try:
            losses = load_losses(path)
        except FileNotFoundError:
            histories[model_name] = None
            print(f"{MODEL_LABELS[model_name]}: no history at {path}")
        except (OSError, json.JSONDecodeError, ValueError) as error:
            histories[model_name] = None
            print(
                f"{MODEL_LABELS[model_name]}: could not load history: {error}",
                file=sys.stderr,
            )
        else:
            histories[model_name] = losses
            epochs, _training_losses, validation_losses = losses
            best_index = min(
                range(len(validation_losses)),
                key=validation_losses.__getitem__,
            )
            print(
                f"{MODEL_LABELS[model_name]}: loaded {len(epochs)} epochs; best validation "
                f"loss {validation_losses[best_index]:.4f} at epoch "
                f"{epochs[best_index]}."
            )

    if not any(losses is not None for losses in histories.values()):
        print("error: no model histories could be loaded", file=sys.stderr)
        return 1

    try:
        figure, select_tab, initial_model = plot_loss_tabs(histories)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.save is not None:
        base_path = args.save.resolve()
        base_path.parent.mkdir(parents=True, exist_ok=True)
        for model_name in MODEL_NAMES:
            if histories[model_name] is None:
                continue
            select_tab(model_name)
            output_path = _model_output_path(base_path, model_name)
            figure.savefig(output_path, dpi=160)
            print(f"Saved {model_name} plot to {output_path}")
        select_tab(initial_model)

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
