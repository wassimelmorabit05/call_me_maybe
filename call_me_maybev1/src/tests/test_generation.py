# ABOUTME: End-to-end test of generate_function_call() using a small scripted fake model
# ABOUTME: instead of a real LLM, so the full template+masking+stopping logic is exercised.

from src.generation import _build_instruction_prompt, generate_function_call
from src.models import FunctionDefinition, ParameterSchema, ReturnsSchema


class _FakeTensor:
    """Stand-in for the torch.Tensor that Small_LLM_Model.encode() normally returns."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[list[int]]:
        return [self._ids]


class ScriptedModel:
    """Fake model that "knows" the correct final text and always scores the token
    that moves generation toward it the highest.

    This exercises the real masking/stopping machinery in generation.py without
    depending on a real LLM, real weights, or network access.
    """

    def __init__(self, full_target_text: str, token_text_cache: dict[int, str]) -> None:
        self._target = full_target_text
        self._cache = token_text_cache
        self._last_text = ""

    def encode(self, text: str) -> _FakeTensor:
        self._last_text = text
        return _FakeTensor([0])  # content is unused; get_logits_from_input_ids ignores it

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        remaining = self._target[len(self._last_text):]
        vocab_size = max(self._cache) + 1
        logits = [0.0] * vocab_size
        for token_id, text in self._cache.items():
            if text and remaining.startswith(text):
                logits[token_id] = float(len(text))  # longer correct matches score higher
        return logits


def _function_definitions() -> list[FunctionDefinition]:
    return [
        FunctionDefinition(
            name="fn_add_numbers",
            description="Add two numbers together.",
            parameters={
                "a": ParameterSchema(type="number"),
                "b": ParameterSchema(type="number"),
            },
            returns=ReturnsSchema(type="number"),
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Greet a person by name.",
            parameters={"name": ParameterSchema(type="string")},
            returns=ReturnsSchema(type="string"),
        ),
    ]


def test_generate_function_call_add_numbers() -> None:
    prompt = "What is the sum of 2 and 3?"
    functions = _function_definitions()

    instruction = _build_instruction_prompt(prompt, functions)
    full_target = instruction + '\n{"name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}'

    token_text_cache = {
        0: "fn_add_numbers",
        1: "fn_greet",
        2: '"',
        3: "2",
        4: "3",
        5: ",",
        6: "}",
    }
    model = ScriptedModel(full_target, token_text_cache)

    result = generate_function_call(prompt, model, token_text_cache, functions)

    assert result.name == "fn_add_numbers"
    assert result.parameters == {"a": 2.0, "b": 3.0}
    assert result.prompt == prompt


def test_generate_function_call_greet() -> None:
    prompt = "Greet john"
    functions = _function_definitions()

    instruction = _build_instruction_prompt(prompt, functions)
    full_target = instruction + '\n{"name": "fn_greet", "parameters": {"name": "john"}}'

    token_text_cache = {
        0: "fn_add_numbers",
        1: "fn_greet",
        2: '"',
        3: "j",
        4: "o",
        5: "h",
        6: "n",
        7: "}",
    }
    model = ScriptedModel(full_target, token_text_cache)

    result = generate_function_call(prompt, model, token_text_cache, functions)

    assert result.name == "fn_greet"
    assert result.parameters == {"name": "john"}