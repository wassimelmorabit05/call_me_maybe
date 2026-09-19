*This project has been created as part of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].*

# call me maybe — Introduction to function calling in LLMs

## Description

This project translates natural-language prompts into structured function
calls (name + typed arguments) using a small local LLM (`Qwen/Qwen3-0.6B`).
Instead of asking the model to answer the question directly, the model is
used to pick *which* function to call and to extract *what* the arguments
are, and the output is guaranteed to be 100% valid, schema-compliant JSON
through constrained decoding rather than relying on the model's own
willingness to produce clean JSON.

## Instructions

```bash
# 1. Copy the provided llm_sdk package next to this README (already done here)
# 2. Install dependencies
make install          # runs `uv sync`

# 3. Run on the default input files (data/input/*.json)
make run               # runs `uv run python -m src`

# or with custom paths:
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

# 4. Lint / type-check
make lint
make lint-strict

# 5. (optional, not graded) run the deterministic unit tests
uv run pytest
```

> Note: `Small_LLM_Model.get_path_to_vocab_file()` downloads the tokenizer
> vocabulary from the Hugging Face Hub on first run, so an internet
> connection is required at least once.

## Algorithm explanation

Constrained decoding is applied at the field level rather than generating
raw JSON syntax token-by-token:

1. **Function name** — the model is prompted with the list of available
   functions and their descriptions. Generation proceeds token by token;
   at every step, the raw logits are masked so that only tokens which keep
   the partial string a valid **prefix** of one of the known function
   names can be selected (`vocab.ids_matching_prefix`). Every other logit
   is set to `-inf` before `argmax` runs. This guarantees the chosen name
   is always one of the functions declared in `functions_definition.json`
   — never a hallucinated name — and the choice is made entirely by the
   model's own scores, never by a heuristic.
2. **Each parameter**, in declaration order, is generated with a small
   dedicated prompt ("give the exact value for parameter X"), masked
   according to its declared type:
   - `number`: only digit tokens (plus a single leading `-` and any `.`)
     are allowed; generation stops once the model's own unmasked top
     choice stops looking numeric.
   - `boolean`: constrained to the two-candidate set `{"true", "false"}`
     using the same prefix-masking mechanism as function names.
   - `string`: generated with ordinary greedy decoding (argmax) until a
     stop token or the token limit is reached. Free text does not have a
     structural constraint to enforce, and any character it produces
     (quotes, newlines, unicode) is safely escaped by `json.dumps` when
     the final object is serialized.
3. **Assembly** — `prompt`, `name`, and `parameters` are assembled into a
   `pydantic` model and serialized with `json.dumps`. The JSON syntax
   itself (braces, quotes, commas) is therefore never something the LLM
   has to get right — it is produced by Python — while every piece of
   *content* inside that JSON (which function, which arguments) is chosen
   by the LLM under a schema-derived mask.

See `src/generator.py` for the implementation (`_select_from_candidates`,
`_extract_number`, `_extract_string`, `_extract_boolean`) and `src/vocab.py`
for how the vocabulary file is indexed (`digit_token_ids`, `dot_token_ids`,
`minus_token_ids`, `ids_matching_prefix`).

## Design decisions

- **Field-level masking instead of a full JSON grammar state machine.**
  Enforcing "which function name" and "is this a valid number" already
  covers every place the schema imposes a real constraint. Hand-rolling a
  full character-level JSON parser/masker for arbitrary nesting would add
  a lot of surface area for bugs without changing the guarantee: syntactic
  validity comes from `json.dumps`, not from the model.
- **`pydantic` models mirror the two input schemas and the output schema**,
  so malformed input files fail fast with a clear validation error instead
  of an obscure `KeyError` deep in the generation loop.
- **Per-prompt error isolation** — `__main__.py` catches exceptions around
  a single prompt and logs a warning instead of aborting the whole batch,
  so one bad/ambiguous prompt cannot take down the run.
- **Vocabulary indexes are built once** at startup (`Vocabulary._build_indexes`)
  rather than re-scanned on every generation step, to keep the per-token
  masking cost low.

## Performance analysis

- **Accuracy**: function selection is bounded to the declared function
  names by construction (cannot select an unknown name), and each
  parameter's type is enforced by construction (a `number` field can never
  come back as free text). Semantic correctness (picking the *right*
  function/value for a given prompt) still depends on the base model's
  language understanding.
- **JSON validity**: 100% by construction — the model never emits raw
  JSON syntax; only field content is generated, then serialized with
  `json.dumps`.
- **Speed**: dominated by the number of forward passes, which is
  `O(tokens generated per field)`. Numeric and boolean fields typically
  resolve in 1-3 tokens; function-name selection and string fields are
  capped (`MAX_NAME_TOKENS`, `MAX_STRING_TOKENS`, ...) to bound worst-case
  latency per prompt.

## Challenges faced

- The SDK only exposes logits for the **next** token given the full
  `input_ids` so far (no KV-cache reuse is exposed), meaning every
  generated token requires a fresh forward pass over the whole growing
  sequence — this is the main cost driver and the reason field values are
  kept short and bounded rather than generating a large free-form JSON
  document.
- Tokenizer leading-space markers (`Ġ` / `▁`) had to be normalized before
  any prefix comparison against plain strings like function names would
  work correctly (see `Vocabulary.clean`).

## Testing strategy

- `tests/test_core.py` covers everything that does not require loading
  the actual LLM: pydantic schema validation (accepts valid input, rejects
  an invalid `type`), output file writing (including that special
  characters inside a string parameter still round-trip through valid
  JSON), and the vocabulary prefix-matching logic against a small fake
  vocabulary.
- End-to-end testing (full pipeline with the real model) was done manually
  by running `make run` against `data/input/function_calling_tests.json`
  and inspecting `data/output/function_calling_results.json`, including
  edge cases: ambiguous prompts, larger numbers, and functions with
  multiple string parameters (`fn_substitute_string_with_regex`).

## Example usage

```bash
uv run python -m src
cat data/output/function_calling_results.json
```

Expected shape for one entry:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Resources

- [Hugging Face — Generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Guidance / grammar-constrained decoding overview](https://github.com/guidance-ai/guidance)
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- Byte Pair Encoding (BPE) — original paper: Sennrich et al., *Neural
  Machine Translation of Rare Words with Subword Units* (2016)

### How AI was used

AI (Claude) was used to help design the overall architecture (splitting
constrained decoding into per-field masking rather than a full JSON
grammar), to draft the initial versions of `src/vocab.py`,
`src/generator.py`, `src/io_utils.py`, and `src/models.py`, and to write
the first draft of this README. Every generated file was read, understood,
and adjusted by the author(s) before submission, and the generation loop
was manually traced against the SDK's actual method signatures
(`get_logits_from_input_ids`, `get_path_to_vocab_file`, `encode`,
`decode`) to confirm correctness.
