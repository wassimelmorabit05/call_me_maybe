# ABOUTME: Unit tests for the deterministic parts of the project (models,
# ABOUTME: vocabulary indexing, io_utils) that don't require loading the LLM.

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.io_utils import load_functions_definition, load_prompts, write_results
from src.models import FunctionCallResult, FunctionsDefinition, PromptsFile
from src.vocab import Vocabulary

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_functions_path(tmp_path: Path) -> Path:
    data = [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together and return their sum.",
            "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
            "returns": {"type": "number"},
        }
    ]
    path = tmp_path / "functions_definition.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def sample_prompts_path(tmp_path: Path) -> Path:
    data = [{"prompt": "What is the sum of 2 and 3?"}]
    path = tmp_path / "function_calling_tests.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_functions_definition_parses(sample_functions_path: Path) -> None:
    result = load_functions_definition(sample_functions_path)
    assert isinstance(result, FunctionsDefinition)
    assert result.root[0].name == "fn_add_numbers"


def test_prompts_file_parses(sample_prompts_path: Path) -> None:
    result = load_prompts(sample_prompts_path)
    assert isinstance(result, PromptsFile)
    assert result.root[0].prompt == "What is the sum of 2 and 3?"


def test_functions_definition_rejects_bad_type() -> None:
    with pytest.raises(ValidationError):
        FunctionsDefinition.model_validate(
            [{"name": "x", "description": "d", "parameters": {}, "returns": {"type": "banana"}}]
        )


def test_write_results_round_trips(tmp_path: Path) -> None:
    out_path = tmp_path / "out" / "function_calling_results.json"
    results = [
        FunctionCallResult(
            prompt="What is the sum of 2 and 3?",
            name="fn_add_numbers",
            parameters={"a": 2.0, "b": 3.0},
        )
    ]
    write_results(out_path, results)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded == [
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }
    ]


def test_write_results_escapes_string_content(tmp_path: Path) -> None:
    """Even if the model produced a raw quote/newline inside a string
    parameter, json.dumps must still yield parseable JSON."""
    out_path = tmp_path / "out.json"
    results = [
        FunctionCallResult(
            prompt="Reverse 'a\"b'",
            name="fn_reverse_string",
            parameters={"s": 'a"b\nc'},
        )
    ]
    write_results(out_path, results)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded[0]["parameters"]["s"] == 'a"b\nc'


@pytest.fixture()
def fake_vocab() -> Vocabulary:
    token_to_id = {
        "fn": 0,
        "_add": 1,
        "_numbers": 2,
        "fn_greet": 3,
        "0": 10,
        "1": 11,
        "2": 12,
        ".": 20,
        "-": 21,
        "true": 30,
        "false": 31,
    }
    vocab = Vocabulary(id_to_token={v: k for k, v in token_to_id.items()}, token_to_id=token_to_id)
    vocab._build_indexes()
    return vocab


def test_vocab_digit_index(fake_vocab: Vocabulary) -> None:
    assert fake_vocab.digit_token_ids == {10, 11, 12}
    assert fake_vocab.dot_token_ids == {20}
    assert fake_vocab.minus_token_ids == {21}


def test_vocab_prefix_matching(fake_vocab: Vocabulary) -> None:
    candidates = ["fn_add_numbers", "fn_greet"]
    ids = fake_vocab.ids_matching_prefix(candidates, "")
    # "fn" is a valid first piece of both candidates.
    assert 0 in ids
    # "fn_greet" as a whole token is also a valid first piece.
    assert 3 in ids

    ids_after_fn = fake_vocab.ids_matching_prefix(candidates, "fn")
    assert 1 in ids_after_fn  # "_add" continues "fn_add_numbers"
    assert 3 not in ids_after_fn  # "fn_greet" no longer starts with "fnfn_greet"
