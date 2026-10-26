"""Run the function-calling program from the command line."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

from src.call_generator import BLUE, RESET, generate_function_call
from src.decoding import load_vocabulary
from src.input_loader import validate_functions, validate_prompts


def parse_arguments() -> argparse.Namespace:
    """Read the file paths and optional visualization flag."""
    parser = argparse.ArgumentParser(description="Generate function calls.")
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    parser.add_argument("--visualize", action="store_true",
                        help="Show function and argument generation in color.")
    return parser.parse_args()


def write_results(path: Path, results: list) -> None:
    """Replace the output atomically after writing valid JSON."""
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8",
                                dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(results, file, indent=2, ensure_ascii=True,
                      allow_nan=False)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> None:
    """Validate files, load the SDK, and write calls in input order."""
    functions = validate_functions(args.functions_definition)
    prompts = validate_prompts(args.input)
    output = Path(args.output)
    if output.resolve() in (Path(args.input).resolve(),
                            Path(args.functions_definition).resolve()):
        raise ValueError("The output must not overwrite an input file")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_dir():
        raise ValueError("The output path is a directory")
    results = []
    if prompts:
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("MKL_NUM_THREADS", "4")
        from llm_sdk import Small_LLM_Model
        model = Small_LLM_Model()
        tokens = load_vocabulary(model)
        for index, prompt in enumerate(prompts, 1):
            if args.visualize:
                print(f"\n{BLUE}Prompt {index}/{len(prompts)}:{RESET} "
                      f"{prompt.prompt}", flush=True)
            try:
                result = generate_function_call(prompt, functions,
                                                model, tokens,
                                                visualize=args.visualize)
            except ValueError as error:
                raise ValueError(f"Prompt {index}: {error}") from error
            results.append(result.model_dump())
            if not args.visualize:
                print(f"Processed {index}/{len(prompts)}", flush=True)
    write_results(output, results)
    print(f"Wrote {len(results)} function call(s) to {output}")


def main() -> int:
    """Return a useful exit code and show errors without a traceback."""
    args = parse_arguments()
    try:
        run(args)
    except KeyboardInterrupt:
        print("Error: interrupted by the user", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
