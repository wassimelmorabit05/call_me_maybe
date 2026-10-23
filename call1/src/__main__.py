# ABOUTME: CLI entry point. Wires together loading, the LLM, constrained decoding,
# ABOUTME: and output writing. Run as: uv run python -m src [--functions_definition F] [--input F] [--output F]
#
# WHY THIS FILE EXISTS (read this first):
# This is the "glue" file - it doesn't contain any of the interesting logic
# itself (that's all in the other src/ files). It just, in order:
#   1) reads the command-line arguments
#   2) loads the two input JSON files
#   3) loads the model + vocabulary
#   4) runs generate_function_call() for every prompt
#   5) writes the results to the output JSON file
# Every step is wrapped so that a failure prints a clear message and exits
# cleanly, instead of showing a scary Python traceback to the user.

import argparse
import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.constrained_decoding import build_token_text_cache
from src.generation import FunctionCallResult, GenerationError, generate_function_call
from src.loaders import InputFileError, load_functions_definition, load_test_prompts
from src.visualization import GenerationTracer
from src.vocabulary import VocabFileNotFoundError, load_vocabulary

# These match exactly what the project subject describes as the defaults.
DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line arguments.

    Kept as its own function (instead of inline in main()) so tests can
    call it directly with a custom argv list, without touching sys.argv.
    """
    parser = argparse.ArgumentParser(
        prog="src",
        description="Translate natural-language prompts into structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the natural-language test prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the resulting function calls JSON file.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Print a step-by-step trace of the constrained decoding process.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the whole pipeline, start to finish.

    Returns an exit code on purpose (0 = success, 1 = failure) instead of
    raising, so sys.exit(main()) at the bottom gives a clean process exit
    code without an ugly traceback for expected failure cases.
    """
    args = parse_args(argv)

    # ---- Step 1: load the two input files ----
    # Both can fail for very "normal" reasons (file missing, bad JSON), so
    # we catch our own InputFileError and print something a human can act
    # on, instead of letting a raw FileNotFoundError/JSONDecodeError escape.
    try:
        functions = load_functions_definition(args.functions_definition)
        prompts = load_test_prompts(args.input)
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # ---- Step 2: load the model + build the one-time vocabulary cache ----
    # Loading a real model can fail in many ways we can't all predict in
    # advance (network issues, out of memory, etc.), so the broad
    # `except Exception` here is intentional - this is the one spot in the
    # program where "something from outside our code went wrong" needs a
    # catch-all so we still exit cleanly instead of crashing.
    try:
        model = Small_LLM_Model()
        vocabulary = load_vocabulary(model)
        token_text_cache = build_token_text_cache(model, vocabulary)
    except VocabFileNotFoundError as exc:
        print(f"Error loading the model's vocabulary: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - see comment above
        print(f"Error loading the language model: {exc}", file=sys.stderr)
        return 1

    # ---- Step 3: generate one function call per prompt ----
    results = []
    for test_prompt in prompts:
        if args.visualize:
            print(f'Generating for prompt: "{test_prompt.prompt}"')
        tracer = GenerationTracer(enabled=args.visualize) if args.visualize else None
        try:
            result = generate_function_call(
                test_prompt.prompt, model, token_text_cache, functions, tracer=tracer
            )
        except GenerationError as exc:
            # IMPORTANT: we do NOT just skip this prompt. Grading tools
            # (and our own output file) expect exactly one output entry per
            # input prompt, in the same order. Skipping one would shift
            # every entry after it out of alignment. So on failure we still
            # append a placeholder result (an empty name/parameters) - this
            # one test will simply be marked wrong, instead of silently
            # breaking every test that comes after it.
            print(
                f"Warning: could not generate a function call for prompt "
                f"{test_prompt.prompt!r}: {exc}. Writing an empty placeholder so "
                f"output stays aligned, one entry per input prompt.",
                file=sys.stderr,
            )
            result = FunctionCallResult(prompt=test_prompt.prompt, name="", parameters={})
        results.append(result)

    # ---- Step 4: write the output file ----
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)  # create data/output/ if missing
        with args.output.open("w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results], f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"Error writing output file {args.output}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} function call(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
