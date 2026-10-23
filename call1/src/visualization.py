# ABOUTME: Optional step-by-step visualization of the constrained decoding process.
# ABOUTME: Makes the "mask -> argmax among survivors" mechanism visible, one step at a time.

from dataclasses import dataclass


@dataclass
class GenerationStep:
    """A single recorded decision made during constrained decoding.

    Attributes:
        blank_label: Human-readable name of the blank being filled
            (e.g. "name", or a parameter name like "a").
        text_tail: The last few characters of the text generated so far,
            for context (kept short so traces stay readable).
        num_value_candidates: How many tokens survived masking as valid
            *content* continuations for the current state.
        num_stop_candidates: How many tokens survived masking as the
            *boundary* (stop) option (0 if the blank could not stop yet).
        chosen_text: The actual text of the token argmax selected.
        stopped: True if the model chose to stop the blank at this step.
    """

    blank_label: str
    text_tail: str
    num_value_candidates: int
    num_stop_candidates: int
    chosen_text: str
    stopped: bool


class GenerationTracer:
    """Records and optionally prints a live trace of constrained decoding.

    Pass an instance to generate_function_call() to see, step by step, how
    many tokens were masked out, how many survived, and which one the model
    actually picked - for every single blank in the JSON template.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.steps: list[GenerationStep] = []

    def record(self, step: GenerationStep) -> None:
        """Store a step and, if enabled, print it immediately."""
        self.steps.append(step)
        if self.enabled:
            print(self._format_step(step))

    @staticmethod
    def _format_step(step: GenerationStep) -> str:
        total = step.num_value_candidates + step.num_stop_candidates
        action = "STOP " if step.stopped else "PICK "
        tail = step.text_tail.replace("\n", "\\n")
        return (
            f"  [{step.blank_label:>10}] ...{tail!r:<24} "
            f"candidates={total:>5} "
            f"(value={step.num_value_candidates}, stop={step.num_stop_candidates})  "
            f"{action}-> {step.chosen_text!r}"
        )

    def print_summary(self) -> None:
        """Print a one-line summary after a prompt finishes generating."""
        print(f"  -> {len(self.steps)} total generation step(s)")