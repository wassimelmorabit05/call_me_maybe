*This project has been created as part of the 42 curriculum by oel-mora.*

# Call Me Maybe

## Description

This program turns a request into a function name and its arguments.
It uses Qwen/Qwen3-0.6B through the supplied `llm_sdk`.
It writes function calls as JSON. It does not execute the functions.

## Instructions

Use Python 3.10, 3.11, or 3.12 and install uv.
Run these commands from the project folder:

```bash
uv sync
make run
make lint
```

The first run downloads the model and needs an internet connection.
Later runs can reuse the model cache. Allow several GB for dependencies.

Default files:

- Functions: `data/input/functions_definition.json`
- Requests: `data/input/function_calling_tests.json`
- Results: `data/output/function_calls.json`

The program creates the output folder when needed.
Do not submit the generated output folder or a virtual environment.

## Example usage

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

For the request `What is the sum of 2 and 3?`, a result is:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Project files

- `src/__main__.py`: command line, model loading, and output files.
- `src/input_loader.py`: read and check the input files.
- `src/models.py`: Pydantic models and value validation.
- `src/decoding.py`: vocabulary and token rules.
- `src/call_generator.py`: prompts, function choice, and arguments.
- `llm_sdk/`: the supplied SDK, kept unchanged.

## Algorithm explanation

1. Validate the input files with Pydantic.
2. Read the vocabulary using the SDK's public method.
3. Convert the vocabulary's byte symbols back to UTF-8 bytes.
4. Show the available functions to Qwen in its chat format.
5. Generate a function name, allowing only names from the definitions.
6. Show only the selected function when generating its arguments.
7. Add the known JSON keys and punctuation in Python.
8. For each value, get the model's scores for the next token.
9. Set an invalid token's score to negative infinity and try the next best.
10. Accept a token only if it keeps a valid prefix of the required value.
11. Stop at the value's ending, allowing tokens that also contain punctuation.
12. Validate the values and write the JSON with `json.dump`.

Strings support quotes, backslashes, JSON escapes, and split UTF-8 bytes.
Numbers support negatives, decimals, and exponents. Integers stay integers.
A `number` argument is saved as a float, such as `2.0`.
Booleans, null values, and `enum` choices are also supported.
Array and object arguments are outside this scalar implementation.

## Design decisions

The JSON structure is fixed; the model chooses the values.
No function names or answers from the tests are hardcoded.
Qwen's empty thinking block keeps responses focused on the call.
Only public SDK methods are used. Model libraries stay inside the SDK.
The vocabulary is loaded once. CPU thread defaults avoid excessive threads.
The supplied SDK is excluded from lint; all project source is checked.

## Errors and reliability

Missing files, invalid JSON, duplicate functions, and bad schemas show errors.
A model-selected no-match result also shows an error and exits with code 1.
Generation stops with an error after 256 tokens for one value.
A failed run does not write a fake function or overwrite an existing result.
Successful runs replace the output only after the complete JSON is written.
An empty request array produces `[]` without loading the model.
Valid JSON and valid types do not guarantee the model understood the request.

## Performance analysis

The unchanged moulinette scored 11/11 public and 11/11 private with Qwen.
The final public run took 165 seconds; the private run took 194 seconds.
Both runs used the CPU. The public set also passed on an earlier run.
Speed depends on hardware, prompt length, and the model cache.
The SDK processes the full input for each token, so short prompts help.
These measured results do not guarantee every unseen request is correct.

## Challenges faced

Closing quotes can share a token with commas or braces; both are accepted.
JSON escapes can span tokens; unfinished escapes remain valid prefixes.
Unicode characters can span tokens; bytes are kept until they decode fully.
Unsupported types and failed values now give errors instead of fake calls.

## Testing strategy

`make lint` runs flake8 and mypy with all flags required by the subject.
Separate checks cover decoder boundaries, escapes, numbers, types, and enums.
They also cover bad inputs, safe output writing, docstrings, and type hints.
The test scripts are not part of the submitted project.
To repeat the private moulinette test, place its folder next to this project:

```bash
cd ../moulinette
uv sync
uv run python -m moulinette prepare_exercises --set private
cd ../call4
uv run python -m src --functions_definition ../moulinette/data/input/functions_definition.json --input ../moulinette/data/input/function_calling_tests.json --output ../moulinette/data/input/function_calls.json
cd ../moulinette
uv run python -m moulinette grade_student_answers --set private --student_answer_path data/input/function_calls.json
```

`make run ARGS="--output result.json"` passes custom options to the program.
Use `make bonus` or `uv run python -m src --visualize` for the blue trace.

## Resources

- [Python JSON](https://docs.python.org/3/library/json.html)
- [Pydantic](https://docs.pydantic.dev/latest/)
- [NumPy](https://numpy.org/doc/stable/)
- [Qwen model and chat format](https://huggingface.co/Qwen/Qwen3-0.6B)
- [uv](https://docs.astral.sh/uv/)

AI helped debug and simplify `src/`, configure lint, run checks, and write this
README. Review each part and be ready to explain it during peer evaluation.
