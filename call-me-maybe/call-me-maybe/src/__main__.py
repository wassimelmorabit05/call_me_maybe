# ABOUTME: CLI entry point. Run as:
# ABOUTME:   uv run python -m src [--functions_definition F] [--input F] [--output F]

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generator import generate_function_call
from .io_utils import load_functions_definition, load_prompts, write_results
from .vocab import Vocabulary

try:
    from llm_sdk import Small_LLM_Model
except ImportError as exc:  # pragma: no cover
    print(f"[error] could not import llm_sdk: {exc}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="call-me-maybe",
        description="Translate natural-language prompts into structured function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the test prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to write the results JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model identifier passed to Small_LLM_Model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    functions = load_functions_definition(args.functions_definition).root
    prompts = load_prompts(args.input).root

    if not functions:
        print("[error] functions definition file is empty", file=sys.stderr)
        sys.exit(1)

    try:
        model = Small_LLM_Model(model_name=args.model)
    except Exception as exc:  # noqa: BLE001 - must never crash the run
        print(f"[error] could not load model '{args.model}': {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        vocab = Vocabulary.load(model.get_path_to_vocab_file())
    except Exception as exc:  # noqa: BLE001
        print(f"[error] could not load vocabulary: {exc}", file=sys.stderr)
        sys.exit(1)

    results = []
    for item in prompts:
        try:
            result = generate_function_call(item.prompt, functions, model, vocab)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not stop the batch
            print(f"[warn] failed on prompt '{item.prompt}': {exc}", file=sys.stderr)

    write_results(args.output, results)
    print(f"Wrote {len(results)} result(s) to {args.output}")


if __name__ == "__main__":
    main()
