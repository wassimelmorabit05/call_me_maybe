import argparse

from .decoder import generate_arguments
from .llm_service import create_model, select_function
from .loader import load_functions, load_tests
from .models import FunctionCallResult
from .validator import build_function_call_result
from .vocabulary import load_vocabulary
from .writer import write_results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate function calls from natural-language prompts."
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the function calling pipeline."""

    args = parse_arguments()

    tests = load_tests(args.input)

    functions = load_functions(
        args.functions_definition
    )

    model = create_model()

    vocabulary = load_vocabulary(
        model.get_path_to_vocab_file()
    )

    results: list[FunctionCallResult] = []

    for test in tests:
        function_name = select_function(
            model=model,
            user_prompt=test.prompt,
            functions=functions,
        )

        function = next(
            function
            for function in functions
            if function.name == function_name
        )

        arguments = generate_arguments(
            model=model,
            user_prompt=test.prompt,
            function=function,
            vocabulary=vocabulary,
        )

        result = build_function_call_result(
            user_prompt=test.prompt,
            function=function,
            arguments=arguments,
        )

        results.append(result)

    write_results(
        args.output,
        results,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
