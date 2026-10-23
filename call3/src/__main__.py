import os
import sys
import json
import argparse
from .c_decoding import load_vocabulary
from .generator import generate_function_call
from llm_sdk import Small_LLM_Model  # type: ignore
from .loader import validate_functions, validate_prompts


def main() -> None:
    """
    Parse command-line arguments and execute the function-calling pipeline.

    Validate the input prompts and function definitions, then initialize
    the LLM and load its vocabulary. Generate a function call for each
    valid prompt and save the results to a JSON output file.

    Handle unexpected errors by displaying an error message and exiting.
    """
    green = "\033[32m"
    red = "\033[31m"
    reset = "\033[0m"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json"
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B"
    )
    parser.add_argument(
        "--visualize",
        action="store_true"
    )

    args = parser.parse_args()

    try:
        validate_prom = validate_prompts(args.input)
        validate_func = validate_functions(args.functions_definition)

        model = Small_LLM_Model(model_name=args.model)

        vocab_path = model.get_path_to_vocab_file()
        token_to_id, id_to_token = load_vocabulary(vocab_path)

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results: list[dict] = []
        for i, prompt in enumerate(validate_prom):
            if args.visualize:
                if results:
                    print("\n" + "─" * 100)
                print(f"\n{red}Prompt {i + 1}:{reset} {prompt.prompt}\n")

            data = generate_function_call(
                prompt, validate_func, model, token_to_id, id_to_token,
                visualize=args.visualize
            )
            results.append(data.model_dump())

            if args.visualize:
                print(f"\n{green}Result:{reset}")
                print(json.dumps(data.model_dump(), indent=2))

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        sys.exit(f"[ERROR]: Unexpected error: {e}")


if __name__ == "__main__":
    main()
