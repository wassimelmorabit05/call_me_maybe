# src/visualizer.py
# ABOUTME: Live terminal visualization of the constrained-decoding generation loop.
# ABOUTME: Prints, token by token, how many candidates survive masking and which one was chosen.

from pydantic import BaseModel, ConfigDict


class _Color:
    """ANSI escape codes used for terminal highlighting (no external dependency)."""

    RESET = "\033[0m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"


class GenerationVisualizer(BaseModel):
    """Prints a live, colored trace of the constrained-decoding generation loop.

    Intended to be called from inside the token-by-token generation loop,
    once per generated token, to show which tokens survived the schema
    mask and which one was ultimately selected.

    Attributes
    ----------
    enabled: bool
        If False, all methods become no-ops (useful to silence output
        during automated / non-interactive runs).
    max_preview_chars: int
        How many characters of the generated-so-far text to show per line,
        to keep long JSON outputs readable on one line.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = True
    max_preview_chars: int = 55

    def start_prompt(self, prompt: str) -> None:
        """Print a header announcing a new prompt is being processed."""
        if not self.enabled:
            return
        print(f"\n{_Color.BOLD}▶ Processing:{_Color.RESET} \"{prompt}\"\n")

    def step(
        self,
        step_num: int,
        generated_so_far: str,
        vocab_size: int,
        num_allowed: int,
        chosen_token_str: str,
    ) -> None:
        """Print one line describing a single generation step.

        Parameters
        ----------
        step_num: int
            1-indexed position of this token in the generation.
        generated_so_far: str
            The decoded text accumulated *before* this step's token
            (i.e. what the model had already produced).
        vocab_size: int
            Total number of tokens in the model's vocabulary — the
            denominator, representing "no constraint at all".
        num_allowed: int
            How many token IDs survived the masking at this step —
            the numerator, representing what constrained decoding
            narrowed the choice down to.
        chosen_token_str: str
            The decoded string of the token that was actually selected.
        """
        if not self.enabled:
            return

        preview = generated_so_far.replace("\n", "\\n")
        if len(preview) > self.max_preview_chars:
            preview = "..." + preview[-(self.max_preview_chars - 3):]

        ratio_color = _Color.GREEN if num_allowed <= 5 else _Color.YELLOW
        chosen_display = chosen_token_str.replace("\n", "\\n") or "<empty>"

        print(
            f"Step {step_num:>3} "
            f"| generated: {_Color.DIM}'{preview}'{_Color.RESET} "
            f"| survived masking: {ratio_color}{num_allowed:>6} / {vocab_size}{_Color.RESET} "
            f"| chosen: {_Color.CYAN}'{chosen_display}'{_Color.RESET}"
        )

    def finish(self, final_text: str, is_valid_json: bool, total_steps: int) -> None:
        """Print the final generated text and whether it parsed as valid JSON."""
        if not self.enabled:
            return
        mark = f"{_Color.GREEN}✓" if is_valid_json else f"{_Color.RED}✗"
        label = "Valid JSON" if is_valid_json else "INVALID JSON (this should never happen!)"
        print(f"\n{mark} {label} produced in {total_steps} steps.{_Color.RESET}")
        print(final_text)

# quick manual test — not part of the graded project
if __name__ == "__main__":
    viz = GenerationVisualizer()
    viz.start_prompt("What is the sum of 2 and 3?")

    fake_steps = [
        ("", 3, "{"),
        ("{", 1, "\""),
        ("{\"", 4, "n"),
        ("{\"n", 1, "a"),
        ("{\"na", 1, "m"),
    ]
    vocab_size = 151936
    for i, (so_far, allowed, chosen) in enumerate(fake_steps, start=1):
        viz.step(i, so_far, vocab_size, allowed, chosen)

    viz.finish('{"name": "fn_add_numbers"}', is_valid_json=True, total_steps=len(fake_steps))