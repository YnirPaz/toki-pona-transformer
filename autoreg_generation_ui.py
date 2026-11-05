"""Small Tk UI for interactively sampling a trained Toki Pona model."""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from generation import AutoregressiveGenerator, GenerationStep


MAX_GENERATED_TOKENS = 256
TOP_PROBABILITIES = 5


class GenerationApp:
    def __init__(self, root: tk.Tk, generator: AutoregressiveGenerator) -> None:
        self.root = root
        self.generator = generator
        self.generated_ids: list[int] = []
        self.prompt_at_generation = ""
        self.finished_with_eos = False
        self.cancel_event = threading.Event()
        self.generation_lock = threading.Lock()
        self.events: queue.Queue[tuple[int, str, object]] = queue.Queue()
        self.session = 0
        self.running = False
        self.tooltip: tk.Toplevel | None = None
        self.pending_spell_marker: tuple[str, str] | None = None

        root.title("Toki Pona Transformer")
        root.geometry("850x620")
        root.minsize(620, 440)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        ttk.Label(root, text="Prompt (optional)").grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 4)
        )
        self.prompt = tk.Text(root, height=5, wrap="word", undo=True)
        self.prompt.grid(row=1, column=0, sticky="nsew", padx=14)

        controls = ttk.Frame(root)
        controls.grid(row=2, column=0, sticky="ew", padx=14, pady=10)
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Temperature").grid(row=0, column=0, padx=(0, 8))
        self.temperature = tk.DoubleVar(value=1.0)
        ttk.Scale(
            controls,
            from_=0.1,
            to=2.0,
            variable=self.temperature,
            command=self._update_temperature_label,
        ).grid(row=0, column=1, sticky="ew")
        self.temperature_label = ttk.Label(controls, text="1.00", width=5)
        self.temperature_label.grid(row=0, column=2, padx=(6, 14))
        self.generate_button = ttk.Button(
            controls, text="Generate to EOS", command=self.generate_to_eos
        )
        self.generate_button.grid(row=0, column=3, padx=3)
        self.one_button = ttk.Button(
            controls, text="Generate 1 token", command=self.generate_one
        )
        self.one_button.grid(row=0, column=4, padx=3)
        self.clear_button = ttk.Button(controls, text="Clear", command=self.clear)
        self.clear_button.grid(row=0, column=5, padx=(3, 0))

        output_frame = ttk.Frame(root)
        output_frame.grid(row=3, column=0, sticky="nsew", padx=14)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)
        ttk.Label(
            output_frame,
            text="Output — hover over generated text to inspect its sampling probabilities",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.output = tk.Text(output_frame, wrap="word", state="disabled", cursor="arrow")
        self.output.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(output_frame, command=self.output.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scrollbar.set)

        self.status = tk.StringVar(value=f"Ready on {generator.device}")
        ttk.Label(root, textvariable=self.status, anchor="w").grid(
            row=4, column=0, sticky="ew", padx=14, pady=(6, 10)
        )

    def _update_temperature_label(self, _value: str = "") -> None:
        self.temperature_label.configure(text=f"{self.temperature.get():.2f}")

    def _set_output(self, text: str) -> None:
        self.pending_spell_marker = None
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", text)
        self.output.configure(state="disabled")

    def _prepare_context(self) -> list[int]:
        prompt = self.prompt.get("1.0", "end-1c")
        if prompt != self.prompt_at_generation:
            self.generated_ids.clear()
            self.finished_with_eos = False
            self.prompt_at_generation = prompt
            prompt_ids = self.generator.encode_prompt(prompt)
            self._set_output(self.generator.tokenizer.decode(prompt_ids))
        return self.generator.encode_prompt(prompt) + self.generated_ids

    def generate_one(self) -> None:
        self._start_generation(1)

    def generate_to_eos(self) -> None:
        self._start_generation(MAX_GENERATED_TOKENS - len(self.generated_ids))

    def _start_generation(self, count: int) -> None:
        if self.running:
            return
        context = self._prepare_context()
        if self.finished_with_eos:
            self.status.set("Generation already reached <EOS>; clear or edit the prompt to restart")
            return
        count = min(count, MAX_GENERATED_TOKENS - len(self.generated_ids))
        if count <= 0:
            self.status.set(f"Stopped at the {MAX_GENERATED_TOKENS}-token limit")
            return

        temperature = self.temperature.get()
        self.running = True
        cancel_event = threading.Event()
        self.cancel_event = cancel_event
        self.generate_button.configure(state="disabled")
        self.one_button.configure(state="disabled")
        self.status.set("Generating…")
        session = self.session

        def worker() -> None:
            try:
                with self.generation_lock:
                    for step in self.generator.generate(
                        context,
                        temperature=temperature,
                        max_new_tokens=count,
                        top_count=TOP_PROBABILITIES,
                    ):
                        if cancel_event.is_set():
                            break
                        self.events.put((session, "token", step))
                self.events.put((session, "done", None))
            except Exception as error:  # surfaced in the UI, where it is actionable
                self.events.put((session, "error", error))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(20, self._poll_events)

    def _poll_events(self) -> None:
        keep_polling = self.running
        while True:
            try:
                session, event, value = self.events.get_nowait()
            except queue.Empty:
                break
            if session != self.session:
                continue
            if event == "token":
                assert isinstance(value, GenerationStep)
                self._append_step(value)
            elif event == "error":
                self.running = False
                self.status.set(f"Generation failed: {value}")
                self._enable_generation_buttons()
            elif event == "done":
                self.running = False
                if self.finished_with_eos:
                    self.status.set(
                        f"Reached <EOS> after {len(self.generated_ids)} generated tokens"
                    )
                elif len(self.generated_ids) >= MAX_GENERATED_TOKENS:
                    self.status.set(f"Stopped at the {MAX_GENERATED_TOKENS}-token limit")
                else:
                    self.status.set(f"Generated {len(self.generated_ids)} token(s)")
                self._enable_generation_buttons()
        if keep_polling and self.running:
            self.root.after(20, self._poll_events)

    def _enable_generation_buttons(self) -> None:
        self.generate_button.configure(state="normal")
        self.one_button.configure(state="normal")

    def _append_step(self, step: GenerationStep) -> None:
        self.generated_ids.append(step.token_id)
        if step.token_id == self.generator.tokenizer.eos_id:
            self.finished_with_eos = True
        if (
            step.token_id == self.generator.tokenizer.end_id
            and self._finish_spelling()
        ):
            return
        display = step.text_piece
        if not display:
            display = f"‹{step.token.strip('<>')}›"
        tag = f"generated_{len(self.generated_ids)}"
        self.output.configure(state="normal")
        start = self.output.index("end-1c")
        self.output.insert("end", display, (tag, "generated"))
        if step.token_id == self.generator.tokenizer.spell_id:
            self.pending_spell_marker = (start, self.output.index("end-1c"))
        self.output.tag_configure("generated", background="#eef5ff")
        self.output.configure(state="disabled")
        self.output.see("end")
        tooltip_text = self._tooltip_text(step)
        self.output.tag_bind(
            tag,
            "<Enter>",
            lambda event, text=tooltip_text: self._show_tooltip(event, text),
        )
        self.output.tag_bind(tag, "<Leave>", self._hide_tooltip)

    def _finish_spelling(self) -> bool:
        """Collapse a completed tokenized spelling to its decoded text."""
        if self.pending_spell_marker is None:
            return False
        start, end = self.pending_spell_marker
        self._hide_tooltip()
        self.output.configure(state="normal")
        self.output.delete(start, end)
        self.output.configure(state="disabled")
        self.output.see("end")
        self.pending_spell_marker = None
        return True

    @staticmethod
    def _tooltip_text(step: GenerationStep) -> str:
        lines = [
            f"Selected: {step.token!r}  {step.probability:.2%}",
            "",
            "Top candidates:",
        ]
        lines.extend(
            f"{candidate.token!r}: {candidate.probability:.2%}"
            for candidate in step.top_probabilities
        )
        if all(candidate.token_id != step.token_id for candidate in step.top_probabilities):
            lines.extend(("", f"Selected token: {step.probability:.2%}"))
        return "\n".join(lines)

    def _show_tooltip(self, event: tk.Event[tk.Misc], text: str) -> None:
        self._hide_tooltip()
        tooltip = tk.Toplevel(self.root)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 12}")
        ttk.Label(
            tooltip,
            text=text,
            justify="left",
            padding=8,
            relief="solid",
            borderwidth=1,
        ).pack()
        self.tooltip = tooltip

    def _hide_tooltip(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        if self.tooltip is not None:
            self.tooltip.destroy()
            self.tooltip = None

    def clear(self) -> None:
        self.cancel_event.set()
        self.session += 1
        self.running = False
        self.generated_ids.clear()
        self.prompt_at_generation = ""
        self.finished_with_eos = False
        self.prompt.delete("1.0", "end")
        self._set_output("")
        self._hide_tooltip()
        self._enable_generation_buttons()
        self.status.set(f"Ready on {self.generator.device}")


def parse_args() -> argparse.Namespace:
    project_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=project_directory / "runs" / "medium-do0.1-v2",
        help="run directory containing tokenizer.json (default: runs/medium-do0.1-v2)",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="best, latest, or a checkpoint path relative to the run directory (default: best)",
    )
    parser.add_argument("--device", help="PyTorch device (default: cuda when available, else cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = AutoregressiveGenerator.load(
        args.run,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    root = tk.Tk()
    GenerationApp(root, generator)
    root.mainloop()


if __name__ == "__main__":
    main()
