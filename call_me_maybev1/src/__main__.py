# ABOUTME: CLI entry point. Wires together loading, the LLM, constrained decoding,
# ABOUTME: and output writing. Run as: uv run python -m src [--functions_definition F] [--input F] [--output F]

import argparse
import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.constrained_decoding import build_token_text_cache
from src.generation import GenerationError, generate_function_call
from src.loaders import InputFileError, load_functions_definition, load_test_prompts
from src.vocabulary import VocabFileNotFoundError, load_vocabulary

DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse; defaults to sys.argv when None.

    Returns:
        The parsed arguments namespace.
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline: load inputs, generate function calls, write output.

    Args:
        argv: Optional argument list (for testing); defaults to sys.argv.

    Returns:
        Process exit code: 0 on success, 1 on any fatal error.
    """
    args = parse_args(argv)

    try:
        functions = load_functions_definition(args.functions_definition)
        prompts = load_test_prompts(args.input)
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        model = Small_LLM_Model()
        vocabulary = load_vocabulary(model)
        token_text_cache = build_token_text_cache(model, vocabulary)
    except VocabFileNotFoundError as exc:
        print(f"Error loading the model's vocabulary: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # model loading can fail in many unpredictable ways
        print(f"Error loading the language model: {exc}", file=sys.stderr)
        return 1

    results = []
    for test_prompt in prompts:
        try:
            result = generate_function_call(
                test_prompt.prompt, model, token_text_cache, functions
            )
        except GenerationError as exc:
            print(
                f"Warning: skipping prompt {test_prompt.prompt!r}: {exc}",
                file=sys.stderr,
            )
            continue
        results.append(result)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results], f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"Error writing output file {args.output}: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(results)} function call(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())