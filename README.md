*This project has been created as part of the 42 curriculum by oel-mora.*

# Description

`call_me_maybe` is a Python project that uses a Large Language Model (LLM) to generate function calls from natural-language requests.

The main goal is to generate valid function names and parameters using **constrained decoding**.

Instead of allowing the model to generate any token, the program limits its choices to tokens that respect the expected data type and function definition.

The project supports:
- Function name selection.
- String generation.
- Number and integer generation.
- Boolean generation.
- JSON-compatible function call results.

The default model is `Qwen/Qwen3-0.6B`.

# Instructions

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Internet connection for the model download.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd call_me_maybe
```

Install dependencies:

```bash
uv sync
```

## Execution

Run the project:

```bash
uv run python -m src
```

Run with visualization:

```bash
uv run python -m src --visualize
```

Use custom input and output files:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The generated results are saved in `data/output/function_calls.json` by default.

# Algorithm Explanation

The project uses constrained decoding to control the model's output.

At each generation step:

1. The program gets the current text.
2. It finds all valid next tokens.
3. Invalid tokens are removed by masking their logits with negative infinity (`-inf`).
4. The model selects the best remaining token.
5. The selected token is added to the generated text.
6. The process continues until a valid stopping condition is reached.

Different generation functions handle different types:

- **Choice:** Selects a function name or a Boolean value from a fixed list.
- **String:** Checks that generated characters do not break string escaping.
- **Number:** Allows valid numeric values.
- **Integer:** Prevents decimal points.

This approach helps generate values that respect the function definitions.

# Design Decisions

- **Separate modules:** Input loading, decoding, models, and generation are kept in different files.
- **Token cache:** Token IDs are mapped to their text to avoid decoding the vocabulary repeatedly.
- **Type-specific generation:** Each parameter is generated according to its declared type.
- **Full-context encoding:** The complete generated context is encoded at each step.
- **Maximum token limit:** Each value has a limit of 64 generation steps.
- **Error handling:** A failed function call produces a warning and an invalid placeholder result instead of stopping the whole prompt-processing loop.

# Performance Analysis

The solution improves structural reliability by restricting the model to valid token choices.

- **Accuracy:** Depends on the model's understanding of the user's request. Valid output does not guarantee the correct function or argument values.
- **Speed:** The model is called at each generation step. Re-encoding the full context can increase processing time.
- **Reliability:** Token restrictions reduce invalid values, while error handling allows other prompts to continue after a failed generation.

No formal speed or accuracy benchmark is provided.

# Challenges Faced

- **Invalid JSON:** Solved by generating values inside a fixed JSON structure.
- **Invalid parameter types:** Solved by using separate generation rules for strings, numbers, integers, and Booleans.
- **Invalid token selection:** Solved by masking tokens that do not respect the current constraints.
- **Infinite or repeated generation:** Controlled with a maximum token limit and string repetition detection.
- **Generation errors:** Handled with warnings and placeholder results.

# Testing Strategy

The implementation can be validated with the following checks:

- Test function selection with different prompts.
- Test valid and invalid numeric prefixes.
- Test integer generation without decimal points.
- Test strings containing quotes and backslashes.
- Test both `true` and `false` Boolean values.
- Test unsupported parameter types and error handling.
- Check that the output JSON contains one result per input prompt.
- Run the application with visualization to inspect token selection.

# Example Usage

Input prompt:

```json
[
  {
    "prompt": "Add 2 and 3"
  }
]
```

Example function definition:

```json
[
  {
    "name": "add",
    "description": "Add two numbers",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

Example generated result:

```json
[
  {
    "prompt": "Add 2 and 3",
    "name": "add",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

# Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Hugging Face Qwen Models](https://huggingface.co/Qwen)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Python Documentation](https://docs.python.org/3/)

## AI Usage

AI was used as a learning and debugging assistant during the project.

It was used to help understand concepts such as function calling, tokens, logits, constrained decoding, JSON validation, and Python type hints. It was also used to discuss errors and possible debugging approaches.

The implementation was reviewed, tested, and understood before being included in the project.