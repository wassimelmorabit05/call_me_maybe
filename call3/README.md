*This project has been created as part of the 42 curriculum by ahamouda.*

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

Call Me Maybe is a Python project that uses a small Large Language Model (LLM) to turn natural-language requests into structured function calls. The program reads a list of user prompts and available function definitions, selects a suitable function, generates its arguments, and writes the results to a JSON file.

For example, given:

```text
What is the sum of 40 and 2?
```

A successful result would look like:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {"a": 40.0, "b": 2.0}
}
```

The main learning objective is **constrained decoding**: instead of freely accepting every predicted token, the program restricts token selection while the model generates a function name and numeric argument values.

---

## Instructions

### Requirements

- Python 3.10 or later
- The `uv` package manager
- The `llm_sdk` supplied for the project, with its model files available
- Project dependencies installed through `uv`

### Installation

```bash
# Clone your repository
git clone <your_repository_url>
cd <repository_directory>

# Place the provided llm_sdk package in the project root
cp -r /path/to/llm_sdk .

# Install dependencies
uv sync
```

Replace the placeholders with the locations used on your machine. The SDK is provided separately for this project.

### Running The Program

```bash
# Use the default input and output paths
uv run python -m src.main

# Specify your own paths
uv run python -m src.main \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

By default, the program reads the function definitions and prompts from `data/input/` and writes its results to `data/output/function_calls.json`.

### Debug Mode

You can run the entry point with Python's debugger:

```bash
uv run python -m pdb src/main.py
```

This allows you to inspect the generated token IDs, selected function name, and parameter values during execution.

### Lint

If `flake8` and `mypy` are installed in your project environment, run:

```bash
uv run flake8 src
uv run mypy src
```

These tools help detect style violations and potential type errors; running them does not itself guarantee that the project passes every check.

### Clean

Remove temporary Python cache directories and any output files you no longer need. If your repository has a Makefile with a `clean` target, you can use:

```bash
make clean
```

The availability of this command depends on your Makefile.

### Tests

If you have created a `tests/` directory and installed `pytest`, run:

```bash
uv run pytest tests/ -v
```

Test coverage and passing results should be confirmed locally before reporting them.

---

## Algorithm Explanation

### Constrained Decoding

A language model predicts a score, called a **logit**, for each possible next-token ID. The program then controls which tokens are eligible for selection:

1. Give the LLM the input context and tokens generated so far.
2. Retrieve the model's next-token logits.
3. Determine which token IDs are valid for the current generation step.
4. Replace the logits of invalid tokens with `-inf`.
5. Select the highest-scoring remaining token and repeat as needed.

Logits are raw prediction scores, not probabilities. In this implementation, `np.argmax()` selects the highest-scoring allowed token.

### Generation Pipeline

```text
User prompt + function definitions
               |
               v
       Build model prompt
               |
               v
        Encode to token IDs
               |
               v
       LLM predicts logits
               |
               v
     Find valid next-token IDs
               |
               v
         Mask invalid logits
               |
               v
         Select next token
               |
               v
    Append token and repeat
               |
               v
   Function name -> parameters
               |
               v
      Write JSON output file
```

Function-name selection uses constrained decoding. Numeric values are generated with a separate set of allowed number and stopping tokens; string values use a different generation loop.

### Prefix Matching

The function name is generated incrementally. At every step, the program compares the current prefix with the names of available functions:

1. Begin with an empty prefix, `""`.
2. Keep only function names starting with that prefix.
3. Check which vocabulary tokens can extend the prefix while remaining the beginning of an allowed function name.
4. Mask out all other tokens, then let the LLM select among the valid ones.
5. Stop when the generated text matches an available function name.

For example:

```text
current_prefix = "fn_"
valid names    = ["fn_add_numbers", "fn_add_strings"]

Possible next tokens might include "add", "add_", or "add_numbers".
A token such as "greet" is rejected because "fn_greet" is not allowed here.
```

The vocabulary search determines what is **valid**; the model's logits determine what it **selects**. This restricts the output to available names but does not guarantee that the chosen function is the best semantic match.

### Schema Enforcement

The program constructs function-call results with the fields `prompt`, `name`, and `parameters`:

```json
{
  "prompt": "<original user request>",
  "name": "<selected function name>",
  "parameters": {"<parameter name>": "<generated value>"}
}
```

The generator inserts JSON structural tokens into the model context, and the final Python result is validated by a Pydantic `FunctionCallResult` model before being serialized with `json.dump()`.

The current result model checks the general shape and supported value types. It does not, by itself, prove that every generated argument matches the selected function's exact parameter definition.

---

## Design Decisions

### Why Force Structural Tokens?

Function names and argument values are the parts that require model predictions. Keys such as `name` and `parameters`, together with braces, quotes, colons, and commas, are known beforehand.

The implementation therefore appends several of these structural tokens directly through the vocabulary's token-to-ID mapping. This reduces the amount of JSON syntax that the LLM must generate and makes the generation process easier to control.

### Why Pydantic For Validation?

Pydantic models are used to validate user prompts, function definitions, parameter type declarations, and generated function-call results.

This keeps validation rules close to the data structures and provides useful error information for malformed inputs. The input models also reject unexpected fields where `extra="forbid"` is configured.

### Why Separate `constraints.py`?

In this version of the project, the related helpers are actually stored in **`c_encoding.py`**, not `constraints.py`.

That file loads the vocabulary, finds valid next-token IDs for function-name prefixes, and masks logits. Keeping those operations separate from `generator.py` makes the generation loop easier to read and the helpers easier to test individually.

### Token Constants

Instead of scattering unexplained numeric token IDs throughout the generator, the code looks up token IDs by their vocabulary strings:

```python
input_ids.append(token_to_id['{'])
input_ids.append(token_to_id['"'])
input_ids.append(token_to_id['name'])
```

This makes the purpose of each token clearer. The required strings must exist in the vocabulary used by the model.

---

## Performance Analysis

### Accuracy

The program attempts to select a function from the supplied definitions and generate its arguments. Its accuracy depends on the prompt, the function descriptions, and the model's predicted logits.

No measured accuracy percentages are provided here because a verified benchmark was not supplied. In particular, numeric extraction with regular expressions can be unreliable for some prompts, and string generation is not constrained in the same way as function-name generation.

### Speed

Each generated token requires a call to the LLM for next-token logits. As a result, execution time depends on prompt length, the number and length of arguments, the number of generation steps, and the available hardware.

The project uses the supplied SDK for a small Qwen model. Record actual timing measurements on your machine before publishing speed estimates.

### Reliability

The function-name decoder restricts selection to prefixes of available names. Numeric generation restricts tokens to a numeric-compatible set plus stopping tokens, but it does not validate every intermediate numeric prefix.

The current implementation also has unbounded loops for numeric and string generation, and the Pydantic result model does not fully enforce the chosen function's argument schema. For those reasons, **100% JSON validity, schema compliance, or graceful handling of every edge case should not be claimed without further safeguards and testing**.

---

## Challenges Faced

### 1. Tokenizer Space Encoding

The vocabulary contains tokenizer representations such as `Ġ`, which can represent a leading space. These representations matter when appending token strings and reconstructing generated text.

The string generator currently replaces `Ġ` with an ordinary space before returning a value. Correct token interpretation remains important for prefix matching and numeric-token filtering.

### 2. Token Context Dependency

The model predicts the next token based on the entire input-token sequence, including tokens already generated.

The function-name loop keeps both `gen_ids` (generated token IDs) and `current_function_prefix` (generated name text). The first updates the LLM's context; the second updates the prefix-matching constraints. Both are needed for the intended decoding behavior.

### 3. Infinite Loop Prevention

Numeric and string generation currently use `while True`. They are intended to stop when the model chooses a suitable delimiter: a stopping token for numbers or a token containing a double quote for strings.

Because neither loop has a maximum-token limit, a missing or never-selected stopping token could cause generation to continue indefinitely. Adding a generation limit would be a useful future reliability improvement.

### 4. Negative Numbers

The generator currently checks whether a prompt contains a negative number. If one is found, it extracts numbers from the prompt using a regular expression rather than asking the LLM to generate them.

This is a workaround, not a general solution: extracted numbers may not match argument order, and the extracted values are not added back into the model's token context in that branch.

---

## Testing Strategy

Tests can be organized in a `tests/` directory around the main modules:

- `test_c_encoding.py` — vocabulary loading, valid-next-token selection, and logit masking
- `test_loader.py` — JSON loading, Pydantic input validation, and error handling
- `test_schema.py` — function-call result validation
- `test_generator.py` — prompt construction and number, string, and function-name generation

To run the tests once implemented:

```bash
uv run pytest tests/ -v
```

Important cases to check include:

- Missing files and invalid JSON
- Empty or malformed prompts
- Duplicate or invalid function definitions
- Function-name prefixes with multiple valid continuations
- Negative numbers, decimals, and malformed numeric tokens
- Missing stopping tokens or excessively long generations

This list describes recommended coverage; it is not a claim that all these tests currently exist or pass.

---

## Example Usage

### Basic Run

From the repository root:

```bash
uv run python -m src.main
```

### Custom Input/Output

```bash
uv run python -m src.main \
  --functions_definition my_functions.json \
  --input my_prompts.json \
  --output results.json
```

### Input Format

`functions_definition.json`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

`function_calling_tests.json`:

```json
[
  {"prompt": "What is the sum of 3 and 7?"},
  {"prompt": "What is the sum of 40 and 2?"}
]
```

### Output Format

`function_calls.json` contains a list of generated results. A successful example is:

```json
[
  {
    "prompt": "What is the sum of 3 and 7?",
    "name": "fn_add_numbers",
    "parameters": {"a": 3.0, "b": 7.0}
  },
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {"a": 40.0, "b": 2.0}
  }
]
```

The example illustrates the expected format; actual values depend on the model's output.

---

## Resources

### Official Documentation

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python argparse](https://docs.python.org/3/library/argparse.html)
- [pytest Documentation](https://docs.pytest.org/)
- [flake8 Documentation](https://flake8.pycqa.org/)
- [mypy Documentation](https://mypy.readthedocs.io/)

### Articles and Research

- [Constrained Decoding for LLMs](https://huggingface.co/blog/constrained-beam-search)
- [Outlines: Structured Generation](https://github.com/outlines-dev/outlines)
- [JSON Schema](https://json-schema.org/)
- [Byte Pair Encoding Tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)

### How AI Was Used

AI assistance was used to explain LLM tokenization, logits, prefix matching, and constrained decoding; discuss implementation problems; review naming; and draft project documentation.

Any statements about implementation ownership, independent development, testing, or verification should reflect the work actually performed and should be confirmed by the project author before submission.
