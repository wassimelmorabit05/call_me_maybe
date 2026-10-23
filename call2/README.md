*This project has been created as part of the 42 curriculum by ael-mora.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling project using a Large Language Model (LLM).

The goal is to convert a natural-language prompt into a structured function call.  
The program uses an LLM to select the appropriate function and generate its arguments while using constrained decoding to ensure that the generated output follows the expected structure and types.

The project generates function calls only. It does not execute the selected functions.

## Instructions

### Installation

Install the project dependencies with:

```bash
make install
```

### Run

Run the program with:

```bash
make run
```

Or directly with:

```bash
uv run python -m src
```

Custom files can be provided with:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Lint

Run code quality and type checks with:

```bash
make lint
```

### Debug

Run the program with the Python debugger:

```bash
make debug
```

### Clean

Remove temporary Python files and caches:

```bash
make clean
```

## Algorithm

The program uses the following pipeline:

1. Load and validate the function definitions and input prompts.
2. Build a prompt describing the available functions.
3. Encode the prompt into token IDs.
4. Ask the LLM for logits for the next token.
5. Use constrained decoding to determine which tokens are valid.
6. Select the valid token with the highest logit.
7. Repeat the process until the function name is generated.
8. Build a new prompt for the selected function's arguments.
9. Generate each argument using constrained decoding according to its expected type.
10. Validate the generated arguments.
11. Write the structured function calls to the output JSON file.

Constrained decoding prevents invalid tokens from being selected during generation. Instead of trusting the LLM to always produce valid output, the decoder restricts generation to tokens that can produce valid function names and argument values.

## Design Decisions

The project is divided into small modules with separate responsibilities:

- `loader.py` loads input JSON files.
- `models.py` defines and validates data structures using Pydantic.
- `prompts.py` builds prompts for the LLM.
- `llm_service.py` creates the model and selects functions.
- `decoder.py` implements constrained decoding and argument generation.
- `validator.py` validates generated arguments.
- `vocabulary.py` loads the model vocabulary.
- `writer.py` writes the final results.
- `__main__.py` coordinates the complete pipeline.

This structure keeps the code easier to understand, test, and maintain.

## Performance Analysis

The implementation is designed to produce valid and schema-compliant function calls.

Function selection is performed by the LLM, while constrained decoding improves reliability by restricting the possible generated tokens.

The solution was tested with different prompts, numeric values, strings, negative numbers, and decimal numbers.

## Challenges Faced

One of the main challenges was understanding token-by-token generation and logits.

Another challenge was implementing constrained decoding so that the LLM could generate useful values while still respecting the required structure and parameter types.

Handling string and number generation separately helped make the decoder easier to control and validate.

## Testing Strategy

The project was tested using:

- Valid function-calling prompts.
- Different function definitions.
- String parameters.
- Number parameters.
- Negative and decimal numbers.
- Invalid JSON input.
- Missing or incorrect parameters.
- Linting with `flake8`.
- Static type checking with `mypy`.

The output is also validated before being written to ensure that generated arguments match the selected function definition.

## Example Usage

Example input prompt:

```text
What is the sum of 2 and 3?
```

Possible generated function call:

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2.0,
        "b": 3.0
    }
}
```

## Resources

Resources used while working on the project:

- Python documentation
- Pydantic documentation
- 42 Call Me Maybe subject
- Provided `llm_sdk`

### AI Usage

AI was used as a learning and debugging assistant during the project.

It was used to help understand concepts such as function calling, tokens, logits, constrained decoding, JSON validation, and Python type hints. It was also used to discuss errors and possible debugging approaches.

The implementation was reviewed, tested, and understood before being included in the project.