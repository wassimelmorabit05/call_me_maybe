# LLM Pipeline: From User Prompt to Structured Output

> A complete walkthrough of how a Large Language Model processes text,
> with every keyword explained — built around the **Call Me Maybe** project.

---

## Table of Contents

1. [Glossary — Every Keyword Explained](#1-glossary--every-keyword-explained)
2. [The Full Pipeline — Step by Step](#2-the-full-pipeline--step-by-step)
   - [Step 1 — User Prompt](#step-1--user-prompt)
   - [Step 2 — Tokenization](#step-2--tokenization)
   - [Step 3 — Input IDs](#step-3--input-ids)
   - [Step 4 — LLM Processing](#step-4--llm-processing)
   - [Step 5 — Logits](#step-5--logits)
   - [Step 6 — Token Selection](#step-6--token-selection)
   - [Step 7 — Autoregressive Loop](#step-7--autoregressive-loop)
3. [Constrained Decoding — The Core of This Project](#3-constrained-decoding--the-core-of-this-project)
4. [How It All Connects to "Call Me Maybe"](#4-how-it-all-connects-to-call-me-maybe)
5. [Visual Summary](#5-visual-summary)

---

## 1. Glossary — Every Keyword Explained

### LLM — Large Language Model

An **LLM** is a neural network trained on massive amounts of text (books, websites,
code, etc.) to predict the next word (or more precisely, the next *token*) in a sequence.

The "large" refers to the number of **parameters** — modern models range from millions
to hundreds of billions of parameters. The model used in this project, **Qwen3-0.6B**,
has ~600 million parameters, making it a *small* LLM.

Training teaches the model statistical patterns in language: after seeing billions of
sentences, it learns that after `"The sky is"`, the token `"blue"` is far more likely
than `"potato"`.

---

### Parameters (Weights)

**Parameters** are the internal numerical values of the neural network — the numbers
that are adjusted during training to make the model smarter.

Think of them as millions of dials. Training turns each dial to the right value so that
when you feed in text, the output makes sense. Once training is done, the parameters
are frozen — inference (using the model) never changes them.

In Qwen3-0.6B: ~600 million parameters → ~600 million floating-point numbers stored
in memory.

```
Input text → [parameter matrix 1] → [parameter matrix 2] → ... → Output scores
```

> **In the project:** You never touch the parameters directly. The `llm_sdk` wraps
> the model and exposes `get_logits_from_input_ids()` which runs the parameters
> internally.

---

### Token

A **token** is the basic unit of text that the model reads and generates. Tokens are
**not** full words — they are *subword* pieces produced by a tokenization algorithm.

Why subwords? Because a vocabulary of whole words would be enormous (millions of
entries), and it would fail on unknown words like names, technical terms, or new slang.
Subword tokenization splits rare words into known pieces.

Examples with **Qwen3** tokenizer:

| Text | Tokens |
|------|--------|
| `"hello"` | `["hello"]` |
| `"unbelievable"` | `["un", "believ", "able"]` |
| `"fn_add_numbers"` | `["fn", "_add", "_numbers"]` |
| `"{"` | `["{"]` |
| `" Paris"` | `["▁Paris"]` (note the leading space marker) |

> **The `▁` (or `Ġ`) prefix** means the token has a space before it in the original
> text. Tokenizers track this to allow perfect reconstruction of the original string.

---

### Vocabulary

The **vocabulary** is the complete, fixed list of all tokens the model knows. Each token
has a unique integer index (its **token ID**).

Qwen3-0.6B has a vocabulary of ~151,936 tokens. The model can only generate tokens
from this list — it cannot invent new tokens.

The `llm_sdk` exposes this via:

```python
model.get_path_to_vocabulary_json()
# Returns path to a JSON file: { "token_string": token_id, ... }
```

> **In the project:** This vocabulary file is **crucial** for constrained decoding.
> It lets you find, for example, all token IDs whose string starts with `"{"` — so
> you can allow only those tokens when the JSON must open with a brace.

---

### BPE — Byte Pair Encoding

**BPE** is the algorithm used to build the vocabulary and to tokenize text.

**How it works (training phase):**
1. Start with individual characters as tokens: `['h','e','l','l','o']`
2. Find the most frequent adjacent pair: e.g., `('e','l')` → merge into `'el'`
3. Repeat until the vocabulary reaches the target size (e.g., 50,000 tokens)

The result is a vocabulary of common character sequences — frequent words become
single tokens, rare words get split into their component pieces.

**How it works (inference phase):**
Given text, apply the learned merge rules in order to produce the token sequence.

> BPE is why `"hello"` might be one token but `"Héllo"` might be two — the accent
> makes it rarer, so it doesn't earn its own merged token.

---

### Tokenization

**Tokenization** is the process of converting a raw text string into a sequence of
tokens using the BPE vocabulary.

```
"What is the sum of 2 and 3?"
        ↓ tokenize
["What", "▁is", "▁the", "▁sum", "▁of", "▁2", "▁and", "▁3", "?"]
```

This is done by `model.encode(text)` in the SDK, which returns a list of integer IDs.

**Important properties of tokenization:**
- It is **deterministic** — the same text always produces the same tokens
- It is **reversible** — tokens can always be decoded back to the original text
- It is **context-free** — each word is tokenized the same way regardless of meaning
- **Numbers are tricky:** `"42"` might be one token, `"1234567"` might be split into
  `["123", "4567"]` — this matters when extracting numeric arguments

---

### Input IDs

**Input IDs** are the integer representations of tokens. After tokenization, each token
is replaced by its index in the vocabulary.

```
["What", "▁is", "▁the", "▁sum", "▁of", "▁2", "▁and", "▁3", "?"]
     ↓ convert to IDs
[3838, 374, 279, 2790, 315, 17, 323, 18, 30]
    (illustrative — actual IDs depend on the model's vocabulary)
```

These integers are what the model actually receives as input. The model knows nothing
about strings — only numbers.

The SDK call: `model.encode(text: str) -> List[int]`

---

### Tensor

A **tensor** is a multi-dimensional array — the fundamental data structure in deep learning.

| Dimensions | Name | Example |
|-----------|------|---------|
| 0D | Scalar | `42` |
| 1D | Vector | `[1, 2, 3]` |
| 2D | Matrix | `[[1, 2], [3, 4]]` |
| 3D | Tensor | `[[[...]]]` |

In the LLM pipeline:
- **Input IDs** are a 1D tensor of shape `(sequence_length,)`
- **Logits** are a 1D tensor of shape `(vocab_size,)` — one score per token in the vocabulary

The SDK uses PyTorch tensors internally. When you call `get_logits_from_input_ids`,
you pass a tensor and receive a tensor.

---

### Embedding

Before the neural network can process token IDs, each integer ID is converted into a
**dense vector** (a list of floating-point numbers). This is called an **embedding**.

```
token_id: 3838  →  embedding: [0.12, -0.87, 0.34, ..., 0.05]  (768 numbers)
```

Each token has a unique embedding vector. These vectors are part of the model's
parameters and encode semantic meaning — similar tokens have similar embedding vectors.

This is the very first operation inside the model, before any computation.

---

### Attention Mechanism (Transformer)

Modern LLMs use the **Transformer** architecture, whose key innovation is the
**attention mechanism**.

Attention lets each token in the sequence "look at" all other tokens and decide which
ones are relevant for predicting the next word.

Example — in `"The animal didn't cross the street because it was too tired"`:
- When processing `"it"`, attention identifies that `"it"` refers to `"animal"`, not `"street"`

In practice: for each token, attention computes a weighted sum of all other token
representations — tokens the model finds more relevant get higher weights.

> **You don't implement this** — it's inside the model. But it's why LLMs understand
> context and long-range dependencies.

---

### Logits

**Logits** are the raw, unnormalized scores the model outputs for each possible next token
after processing the entire input sequence.

```
Shape: (vocab_size,)  →  e.g., (151936,)  — one score per token in the vocabulary

logits[0]    = score for token 0   = -12.3
logits[1]    = score for token 1   =   0.4
logits[42]   = score for token 42  =   8.7   ← high score = likely next token
logits[3838] = score for token 3838=  -4.1
...
```

Logits are not probabilities — they can be any real number (positive or negative). A
higher logit means the model considers that token more likely as the next token.

> **This is the critical output** — the SDK method `get_logits_from_input_ids(input_ids)`
> returns exactly this tensor. **Constrained decoding works by modifying these logits
> before token selection.**

---

### Softmax

**Softmax** is the mathematical function that converts logits into proper probabilities.

```
softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
```

Properties:
- All outputs are between 0 and 1
- All outputs sum to exactly 1
- Relative order is preserved — the highest logit becomes the highest probability

```
logits:        [-12.3,  0.4,  8.7,  -4.1, ...]
after softmax: [~0.00, 0.001, 0.97, ~0.00, ...]
```

In practice, for argmax selection (greedy), you don't even need softmax — you just
pick the index of the maximum logit directly.

---

### Argmax / Greedy Decoding

**Greedy decoding** (also called argmax) always picks the token with the **highest logit**
as the next token.

```python
next_token_id = logits.argmax()
```

It is "greedy" because it makes the locally optimal choice at each step without
considering future steps. This is the simplest and fastest decoding strategy.

**Downside:** greedy decoding can lead to repetitive or suboptimal outputs because the
globally best sequence isn't always built from locally best tokens.

> **In this project**, greedy decoding (argmax over the masked logits) is the correct
> approach — you want deterministic, structured output, not creative variety.

---

### Sampling (Temperature, Top-k, Top-p)

Instead of always picking the max, **sampling** draws a token randomly from the
probability distribution. This introduces controlled randomness.

**Temperature (`T`)** — scales the logits before softmax:
```
scaled_logits = logits / T
```
- `T < 1.0` → sharper distribution → more deterministic (closer to greedy)
- `T > 1.0` → flatter distribution → more random, creative
- `T = 0` → identical to greedy argmax

**Top-k** — only consider the `k` highest-probability tokens when sampling.

**Top-p (nucleus sampling)** — only consider tokens whose cumulative probability
exceeds `p` (e.g., 0.9).

> **In this project**, these are **not relevant** — constrained decoding overrides
> all of this by masking logits directly. You use argmax on the masked logits.

---

### Negative Infinity (`-inf`)

When **masking** logits for constrained decoding, invalid tokens get their logit set to
**negative infinity** (`float('-inf')` or `-numpy.inf`).

Why? After softmax, `exp(-inf) = 0` — so the probability of that token becomes
exactly zero. It will never be selected, regardless of what the model "thinks."

```python
import numpy as np

logits[invalid_token_ids] = -np.inf
next_token = logits.argmax()  # Will never pick masked tokens
```

---

### Context Window

The **context window** (or **context length**) is the maximum number of tokens the
model can process in a single forward pass. Qwen3-0.6B supports up to 32,768 tokens.

Every call to `get_logits_from_input_ids(input_ids)` processes the **entire** current
sequence — prompt + all previously generated tokens. As generation progresses, the
sequence grows by one token per step.

---

### Autoregressive Generation

**Autoregressive** means the model generates one token at a time, feeding each new
token back into the input for the next step.

```
Step 1: input = [prompt tokens]              → generate token_1
Step 2: input = [prompt tokens, token_1]     → generate token_2
Step 3: input = [prompt tokens, token_1, token_2] → generate token_3
...
```

This repeats until a **stop condition** is met (e.g., an EOS token, a `}` closing the
JSON object, or a maximum length).

---

### EOS — End of Sequence

The **EOS token** (End of Sequence) is a special token in the vocabulary that signals
the model has finished generating. When the model selects EOS as the next token, generation stops.

In a constrained decoding setup, you decide when to stop — typically when the JSON
structure is complete (the closing `}` has been generated and the schema is fully satisfied).

---

### JSON Schema

A **JSON schema** defines the expected structure of a JSON object — which keys are
required, what type each value must be, and what values are allowed.

In this project, the schema for each function call is:
```json
{
  "prompt": "<string>",
  "name": "<one of the known function names>",
  "parameters": {
    "<param_name>": <value of correct type>
  }
}
```

The constrained decoder must enforce this schema token by token — never allowing the
model to deviate from it.

---

### Function Calling

**Function calling** is the process of translating a natural language request into a
structured, machine-executable function invocation.

```
Natural language:  "What is the sum of 40 and 2?"
         ↓
Function call:  { "name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0} }
```

The model's job is not to answer the question — it's to identify which function to call
and extract the correct arguments from the text.

---

## 2. The Full Pipeline — Step by Step

### Step 1 — User Prompt

The pipeline begins with a **natural language string** from the user.

```
prompt = "What is the sum of 2 and 3?"
```

Before feeding to the model, this prompt is typically enriched with a **system prompt**
that describes the available functions and instructs the model to produce structured output.

```
system_prompt = """
You are a function-calling assistant. Given a user request,
identify the correct function and extract its arguments.
Available functions:
- fn_add_numbers(a: number, b: number)
- fn_greet(name: string)
- fn_reverse_string(s: string)
"""

full_prompt = system_prompt + "\nUser: " + prompt + "\nOutput JSON:"
```

The richer the system prompt, the better the model's semantic understanding of the
task — but even with a perfect prompt, small models are unreliable without constrained
decoding.

---

### Step 2 — Tokenization

The full prompt string is broken into tokens using the model's **BPE tokenizer**.

```
"What is the sum of 2 and 3?"
         ↓ encode()
["What", "▁is", "▁the", "▁sum", "▁of", "▁2", "▁and", "▁3", "?"]
```

The SDK call:
```python
token_ids: List[int] = model.encode(full_prompt)
```

**What to watch for:**
- Numbers may split unexpectedly: `"265"` could be `["26", "5"]` or `["265"]`
- Punctuation often gets its own token: `"{"` is usually a single token
- String values from the prompt (like `"hello"`) appear as tokens — the constrained
  decoder must reproduce them character by character

---

### Step 3 — Input IDs

Token strings are converted to their integer vocabulary indices.

```
["What", "▁is", "▁the", ...] → [3838, 374, 279, ...]
```

These integers are wrapped in a **tensor** and passed to the model.

```python
import torch
input_ids: torch.Tensor = torch.tensor(token_ids)
```

---

### Step 4 — LLM Processing

The model executes a **forward pass** through its neural network layers:

```
input_ids
    ↓
[Embedding layer]   — converts each ID to a vector (e.g., 1024 floats)
    ↓
[Transformer layer 1]  — attention + feed-forward
    ↓
[Transformer layer 2]
    ↓
   ...                 (Qwen3-0.6B has 28 layers)
    ↓
[Transformer layer 28]
    ↓
[Linear projection]  — projects to vocabulary size
    ↓
logits  (shape: vocab_size)
```

This entire operation happens inside `model.get_logits_from_input_ids(input_ids)`.

**Complexity:** Each forward pass processes the entire sequence. If the prompt is 200
tokens and you've generated 50 tokens so far, the model processes 250 tokens on that
step. This is why generation slows down as the sequence grows.

---

### Step 5 — Logits

The model outputs a **1D tensor of shape `(vocab_size,)`** — one raw score per token.

```python
logits: torch.Tensor = model.get_logits_from_input_ids(input_ids)
# shape: (151936,)

# Example (simplified):
# logits[token_id_for_"{"]     =  12.4   ← model thinks { is likely
# logits[token_id_for_"Hello"] =   0.3
# logits[token_id_for_"42"]    =   8.1
# logits[token_id_for_"xyz"]   =  -9.7
```

At this point, **without constrained decoding**, you would just pick the highest logit.
With constrained decoding, you **modify these logits first**.

---

### Step 6 — Token Selection

#### Without constrained decoding (naive):
```python
next_token_id = int(logits.argmax())
```

This works ~30% of the time for small models producing JSON.

#### With constrained decoding (this project):
```python
# 1. Determine which tokens are valid at this position
valid_token_ids = get_valid_tokens(current_json_state, vocab)

# 2. Mask all invalid tokens
mask = torch.full(logits.shape, float('-inf'))
mask[valid_token_ids] = 0.0
masked_logits = logits + mask

# 3. Pick the best valid token
next_token_id = int(masked_logits.argmax())
```

This works ~100% of the time, regardless of model quality.

---

### Step 7 — Autoregressive Loop

Steps 3–6 repeat until the output is complete.

```python
generated_ids = []

while not is_complete(generated_ids):
    # Build current input = prompt + generated so far
    current_input = torch.tensor(prompt_ids + generated_ids)

    # Forward pass
    logits = model.get_logits_from_input_ids(current_input)

    # Constrained decoding: mask invalid tokens
    valid_ids = get_valid_tokens(current_json_state, vocab)
    logits[invalid_ids] = float('-inf')

    # Select next token
    next_id = int(logits.argmax())

    # Update state
    generated_ids.append(next_id)
    update_json_state(next_id)

# Decode the generated token IDs back to a string
output_text = model.decode(generated_ids)
result = json.loads(output_text)
```

---

## 3. Constrained Decoding — The Core of This Project

### The Problem

Without constraints, a 0.6B model asked to produce JSON might output:

```
"The sum of 2 and 3 is 5, so you should call add_numbers with a=2 and b=3."
```

Or even valid-looking but wrong JSON:
```json
{"function": "add", "args": [2, 3]}
```

Neither matches the required schema.

### The Solution: A JSON State Machine

Constrained decoding works by maintaining a **state** that tracks exactly where you are
in the JSON structure being built. At each step, the state tells you which characters
(and therefore which tokens) are valid next.

#### JSON States for This Project

```
STATE: OPEN
  → only token "{" is valid
  → next state: FIELD_NAME

STATE: FIELD_NAME
  → only tokens for known key names ("prompt", "name", "parameters") are valid

STATE: AFTER_COLON (after "name":)
  → only tokens that start known function names are valid
  → e.g., tokens for "fn_add_numbers", "fn_greet", "fn_reverse_string"

STATE: IN_STRING_VALUE
  → any printable character token is valid, except unescaped quotes
  → closing quote ends the string

STATE: IN_NUMBER_VALUE
  → only digit tokens, ".", "-", "e", "E" are valid
  → comma or "}" ends the number

STATE: CLOSE
  → only "}" is valid
  → generation stops
```

#### How to Find Valid Tokens

The vocabulary JSON maps every token string to its integer ID. To find which tokens are
valid at a given state:

```python
import json

with open(model.get_path_to_vocabulary_json()) as f:
    vocab = json.load(f)
# vocab = {"token_string": token_id, ...}

# Example: find all tokens that could start a function name
valid_tokens = [
    token_id
    for token_str, token_id in vocab.items()
    if any(fn_name.startswith(token_str.lstrip('▁'))
           for fn_name in known_functions)
]
```

#### Schema Enforcement

Constrained decoding enforces **two levels** of validity simultaneously:

1. **Syntactic validity** — the output is always parseable JSON (correct brackets,
   quotes, commas, colons)

2. **Semantic validity** — the content matches the expected schema:
   - `"name"` can only be one of the known function names
   - Parameter values have the correct types (`number` vs `string` vs `boolean`)
   - All required parameters are present
   - No extra keys are added

#### Token Boundary Problem

One subtlety: a token might be multiple characters. When you're building a string like
`"fn_add_numbers"`, the tokenizer might produce:

```
"fn_add_numbers" → ["fn", "_add", "_num", "bers"]
```

Your state machine must handle **partial token matches** — at each step, track what
characters have been generated so far in the current field and determine which tokens
could legally extend the current partial string.

---

## 4. How It All Connects to "Call Me Maybe"

Here is the full data flow of the project:

```
functions_definition.json          function_calling_tests.json
        ↓                                     ↓
[Load & validate with Pydantic]    [Load prompts as list of strings]
        ↓                                     ↓
        └──────────────┬──────────────────────┘
                       ↓
             For each prompt:
                       ↓
          [Build system prompt with
           function definitions]
                       ↓
              [model.encode()]          → token IDs (List[int])
                       ↓
              [Convert to Tensor]       → input_ids (Tensor)
                       ↓
        ┌──────────────────────────────┐
        │     GENERATION LOOP          │
        │                              │
        │  [model.get_logits_from_    │
        │   input_ids(input_ids)]      │ → logits (Tensor, shape: vocab_size)
        │          ↓                   │
        │  [JSON State Machine]        │ → valid_token_ids (List[int])
        │          ↓                   │
        │  [Mask invalid tokens]       │ → logits[invalid] = -inf
        │          ↓                   │
        │  [logits.argmax()]           │ → next_token_id (int)
        │          ↓                   │
        │  [Append to generated_ids]   │
        │  [Update JSON state]         │
        │  [input_ids = concat(        │
        │     input_ids, next_token)]  │
        │          ↓                   │
        │  [Is JSON complete?]──Yes───►│
        └──────────────────────────────┘
                       ↓
              [model.decode(generated_ids)]  → JSON string
                       ↓
              [json.loads()]                 → Python dict
                       ↓
              [Pydantic validation]          → typed result object
                       ↓
              [Append to results list]
                       ↓
             (next prompt)
                       ↓
        [Write results to output JSON file]
                       ↓
        function_calling_results.json  ✓
```

---

## 5. Visual Summary

```
USER PROMPT
"What is the sum of 2 and 3?"
        │
        ▼
TOKENIZATION (BPE)
["What", "▁is", "▁the", "▁sum", "▁of", "▁2", "▁and", "▁3", "?"]
        │
        ▼
INPUT IDs (integers)
[3838, 374, 279, 2790, 315, 17, 323, 18, 30]
        │
        ▼
LLM FORWARD PASS
  Embedding → 28 Transformer layers → Linear projection
        │
        ▼
LOGITS (151936 scores, one per vocab token)
[−12.3, 0.4, 8.7, −4.1, ..., 12.1, ...]
        │
        ▼
CONSTRAINED DECODING (logit masking)
  JSON state machine → valid token set
  logits[invalid tokens] = −∞
        │
        ▼
TOKEN SELECTION (argmax)
  next_token_id = argmax(masked_logits)
        │
        ▼
APPEND & LOOP
  input_ids = concat(input_ids, [next_token_id])
  repeat until JSON is complete
        │
        ▼
DECODE (token IDs → string)
'{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}'
        │
        ▼
OUTPUT (parsed & validated)
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

---

## Key Takeaways

| Concept | One-line summary |
|---------|-----------------|
| LLM | Neural network that predicts the next token in a sequence |
| Parameters | Frozen numerical weights that encode the model's knowledge |
| BPE | Algorithm that splits text into subword tokens |
| Token | The atomic unit of text the model reads and generates |
| Vocabulary | The fixed set of all possible tokens (151,936 for Qwen3) |
| Input IDs | Integer indices mapping tokens to vocabulary positions |
| Tensor | Multi-dimensional array — the data structure for all values |
| Logits | Raw unnormalized scores for every possible next token |
| Softmax | Converts logits to a probability distribution (sums to 1) |
| Argmax | Pick the token with the highest score — greedy decoding |
| -inf masking | Set invalid token logits to −∞ so they get probability 0 |
| Constrained decoding | Modify logits before selection to enforce structure |
| JSON State Machine | Tracks position in JSON to determine which tokens are valid |
| Autoregressive | One token at a time, each new token appended to the input |
| Context window | Maximum total tokens (prompt + generated) the model can handle |
| Function calling | Translating natural language into typed, structured function calls |
