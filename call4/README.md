*This project has been created as part of the 42 curriculum by oel-mora.*

# call me maybe — Function Calling via Constrained Decoding

## Description

This project implements a function-calling tool that translates natural-language
prompts into structured, executable function calls, using a small (0.6B parameter)
local LLM (`Qwen/Qwen3-0.6B` by default, via the provided `llm_sdk` package).

Given a prompt such as *"What is the sum of 40 and 2?"*, the tool does not answer
the question directly. Instead, it outputs the function to call and its typed
arguments:

```json
{"name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

The core challenge this project solves is reliability: small LLMs are notoriously
unreliable at producing syntactically valid, schema-compliant JSON when simply
prompted to do so. This project solves that with **constrained decoding**: at every
generation step, invalid tokens are masked out of the model's logits *before*
selection, making invalid output structurally impossible rather than merely
unlikely.

## Project Structure

```
src/
├── __init__.py        # re-exports the public API
├── __main__.py         # CLI entry point (uv run python -m src) — name required by Python
├── models.py            # FunctionCallResult (output model)
├── input_loader.py       # input validation (Prompt, FunctionDefinition, Parameter)
├── decoding.py             # vocabulary loading + generic numpy masking/argmax
└── call_generator.py       # constrained generation: name/number/integer/string/boolean
```

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/)

### Setup

```bash
make install        # equivalent to: uv sync
```

This installs `numpy` and `pydantic`, and installs `llm_sdk` as a local, editable
dependency (see `pyproject.toml`'s `[tool.uv.sources]`).

### Running

```bash
make run             # equivalent to: uv run python -m src
```

By default, this reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`.

Custom paths, a different model, and a step-by-step trace can be requested directly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json \
  --model Qwen/Qwen3-0.6B \
  --visualize
```

### Other Makefile targets

```bash
make debug        # run under pdb
make lint          # flake8 + mypy (required flags)
make lint-strict   # flake8 + mypy --strict
make test          # run the unit test suite (pytest)
make clean         # remove caches
```

## Resources

- [Hugging Face — Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Hugging Face — Logits processors / constrained generation](https://huggingface.co/docs/transformers/internal/generation_utils)
- [Byte-Pair Encoding (BPE) tokenization — Hugging Face course](https://huggingface.co/learn/nlp-course/chapter6/5)
- [pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html)
- OpenAI / Anthropic function-calling / tool-use documentation (general concept reference)

### How AI was used

AI assistance (Claude) was used throughout this project as a step-by-step design
and pair-programming partner, not as a black box:

- **Scaffolding**: `pyproject.toml` files, directory layout, `Makefile`, `.gitignore`,
  `.flake8`.
- **Design discussion**: the "fixed template + constrained blanks" approach to
  constrained decoding, and the "boundary-token comparison" mechanism used to let
  the model decide when a variable-length value (number/integer/string) should
  stop, were developed interactively with AI and then implemented and verified
  line by line.
- **Code drafting**: initial drafts of `src/models.py`, `src/input_loader.py`,
  `src/decoding.py`, `src/call_generator.py`, `src/__main__.py`, and the unit tests
  in `tests/`.
- **Reviewing a reference implementation**: a separate reference implementation
  (same general approach, different file layout) was reviewed with AI, which
  helped identify and fix several real gaps before submission — unescaped
  quotes/backslashes in string values, no distinct handling of the `"integer"`
  schema type, and unbounded generation loops with no safety cap.
- **Documentation**: this README was drafted with AI assistance.

Every AI-suggested piece of code was read, explained back in detail, and understood
before being accepted, in line with the project's "AI Instructions" chapter — no
code was copy-pasted without being able to justify it.

## Algorithm Explanation

The pipeline treats the required JSON output as a **template with a small number of
blanks**, not as a free-form generation task:

```
{"name": "<BLANK: function name>", "parameters": {"<param>": <BLANK: value>, ...}}
```

All non-blank characters (quotes, braces, commas, key names) are injected directly
as literal text — they are never generated by the model and can therefore never be
wrong.

Four kinds of blanks are supported, each with its own validity rule:

- **Choice** (`generate_choice_value`): the blank must match one of a fixed list of
  known strings — used for the function name, and for boolean values
  (`"true"`/`"false"`). A partial string is valid as long as it is a prefix of at
  least one known choice.
- **Number** (`generate_numeric_value`, schema type `"number"`): a valid (possibly
  partial) JSON number — optional leading `-`, digits, at most one `.`.
- **Integer** (`generate_numeric_value`, schema type `"integer"`): same as above
  but a decimal point is never allowed, guaranteeing a genuine Python `int` in the
  output (required because some functions assert `isinstance(x, int)`).
- **String** (`generate_string_value`): any character, *except* that a literal `"`
  or `\` inside the value must arrive as the standard JSON escape (`\"` or `\\`) —
  a bare, unescaped `"` always ends the string. This is what allows values like a
  regex (`"\\d+"`) or a Windows path (`"C:\\Users\\john"`) to be generated
  correctly, not just plain words.

At each generation step:

1. `get_logits_from_input_ids()` is called to get the raw logits.
2. Every cached vocabulary token is checked against the current blank's validity
   rule. Invalid tokens' logits are set to `-inf` (`decoding.mask_logits`, a
   NumPy array).
3. **Stopping is also decided by the model, not hardcoded**: once a blank's value
   is already complete, the token(s) matching the literal text that follows in the
   template (e.g. `,`, `}`, `"`) are added to the candidate set alongside valid
   continuation tokens. Whichever scores highest via `argmax` wins — so e.g.
   generating `"2"` vs `"265"` for a number is a real model decision, not an
   arbitrary length limit.
4. `select_next_token` (NumPy `argmax`) picks the next token among only the
   surviving candidates.

Because every single generated character was pre-filtered to be both
structurally and semantically valid, the final output is always valid JSON that
matches the function's schema exactly — independent of how "confident" or
"capable" the underlying small model actually is.

## Design Decisions

- **Full re-encoding per step**: rather than concatenating previously-generated
  token IDs with freshly-encoded literal template text, the full accumulated text
  is re-encoded via `model.encode()` at every step. This avoids subtle
  byte-level-BPE boundary mismatches between independently encoded chunks, at the
  cost of one cheap extra `encode()` call per step (negligible next to the
  forward pass itself).
- **Token-text cache built once**: `build_token_text_cache()` decodes every
  vocabulary token exactly once at startup via `model.decode()`, instead of per
  generation step — turning an O(vocab_size × steps) cost into
  O(vocab_size + steps), and incidentally handling BPE's internal space marker
  for free (`decode()` already turns it back into a literal space).
- **`"number"` vs `"integer"` are kept as distinct generation paths**: even though
  both are numeric, only `"number"` ever allows a decimal point. Combined with
  Python's own `json` module (which always serializes a `float` with a decimal
  point), this guarantees `2.0` for a `"number"` field and `2` for an `"integer"`
  field — matching what strict `isinstance()` checks on the calling side expect.
- **No text-scraping heuristics**: every argument value, including negative
  numbers, is produced by the same constrained generation loop — never by
  regex-extracting numbers directly from the user's prompt text. Relying on
  the prompt's wording instead of genuine generation would silently break on
  prompts that don't share that exact phrasing, defeating the purpose of
  constrained decoding.
- **Never drop a prompt**: a value that fails to generate still produces a
  `FunctionCallResult` placeholder (`name="invalid"`, empty `parameters`) rather
  than being skipped, so the output always has exactly one entry per input
  prompt, in order — important for any index-based grading.
- **pydantic throughout**: all input/output schemas (`Prompt`, `FunctionDefinition`,
  `Parameter`, `FunctionCallResult`) are pydantic models (`extra="forbid"` on
  inputs), giving automatic, clearly-erroring validation instead of manual `dict`
  key-checking.

## Performance Analysis

- **Validity**: by construction, every character of the output is either fixed
  template text or a token that survived masking — so output JSON is always
  syntactically valid and always matches the declared schema (100% target met by
  design, not by luck).
- **Accuracy** (which function/values are *semantically* correct) depends on the
  underlying model's judgement when scoring among the valid candidates — masking
  guarantees validity, not correctness.
- **Speed**: the dominant cost per generated character is one forward pass
  (`get_logits_from_input_ids`). The token-text cache (one-time cost) and the
  per-step `encode()` call are comparatively cheap. A `MAX_VALUE_TOKENS` safety
  cap bounds worst-case time per value and prevents runaway loops.

## Challenges Faced

- **Deciding when a variable-length value is "done"**: an early approach of
  stopping as soon as a value became structurally valid (e.g. after a single
  digit) would always truncate multi-digit numbers. Solved by including the
  literal boundary text's token as a real competing candidate in the same
  masked-argmax step, letting the model itself choose between extending or
  stopping.
- **Strings containing quotes or backslashes**: regex and file-path arguments
  (e.g. `\d+`, `C:\Users\john`) need a literal `"` or `\` inside the value. An
  initial design that simply forbade both characters could never produce them.
  Solved by supporting the standard JSON escapes (`\"`, `\\`) during generation
  and unescaping the result afterward.
- **Token/text correspondence under BPE**: concatenating independently-encoded
  text chunks is not guaranteed to retokenize the same way as encoding the whole
  string at once. Resolved by re-encoding the full accumulated text at every
  step rather than concatenating token ID lists.
- **Testing without a real model**: the full generation loop was made testable
  by depending only on the LLM's public interface (`encode`/`decode`/
  `get_logits_from_input_ids`), allowing deterministic "scripted" fake models in
  the test suite.

## Testing Strategy

Unit tests (`tests/`, run via `make test`) cover:

- Input-file loading and validation, including every documented failure mode —
  missing file, invalid JSON, non-array top level, empty array, schema mismatch,
  invalid function name, and duplicate function names (`test_input_loader.py`).
- Masking and selection logic with a fake model (`test_decoding.py`).
- The full `generate_function_call()` loop end-to-end using a scripted fake model
  that deterministically reproduces a known-correct answer — covering negative
  numbers, the `"integer"` type, escaped quotes/backslashes in strings, and
  boolean values (`test_call_generator.py`).

In addition, the pipeline was run manually end-to-end (`make run`) against the
provided `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, with the resulting
`data/output/function_calling_results.json` manually checked for valid JSON and
schema compliance, and against the `moulinette` grading script.

## Example Usage

```bash
make install
make run
cat data/output/function_calling_results.json
```

Example input (`data/input/function_calling_tests.json`):
```json
[{"prompt": "What is the sum of 2 and 3?"}, {"prompt": "Greet john"}]
```

Example output (`data/output/function_calling_results.json`):
```json
[
  {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}},
  {"prompt": "Greet john", "name": "fn_greet", "parameters": {"name": "john"}}
]
```