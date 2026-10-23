# ABOUTME: Unit tests for loaders.py: valid parsing plus every documented failure mode
# ABOUTME: (missing file, invalid JSON, wrong top-level type, schema mismatch).

from pathlib import Path

import pytest

from src.loaders import InputFileError, load_functions_definition, load_test_prompts


def test_load_functions_definition_valid(tmp_path: Path) -> None:
    content = """
    [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers.",
            "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
            "returns": {"type": "number"}
        }
    ]
    """
    path = tmp_path / "functions_definition.json"
    path.write_text(content, encoding="utf-8")

    functions = load_functions_definition(path)

    assert len(functions) == 1
    assert functions[0].name == "fn_add_numbers"
    assert functions[0].parameters["a"].type == "number"


def test_load_test_prompts_valid(tmp_path: Path) -> None:
    path = tmp_path / "function_calling_tests.json"
    path.write_text('[{"prompt": "What is the sum of 2 and 3?"}]', encoding="utf-8")

    prompts = load_test_prompts(path)

    assert len(prompts) == 1
    assert prompts[0].prompt == "What is the sum of 2 and 3?"


def test_missing_file_raises_input_file_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(InputFileError, match="not found"):
        load_functions_definition(missing)


def test_invalid_json_raises_input_file_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('[{"name": "fn_x",}]', encoding="utf-8")  # trailing comma
    with pytest.raises(InputFileError, match="Invalid JSON"):
        load_functions_definition(path)


def test_non_array_top_level_raises_input_file_error(tmp_path: Path) -> None:
    path = tmp_path / "not_an_array.json"
    path.write_text('{"name": "fn_x"}', encoding="utf-8")
    with pytest.raises(InputFileError, match="JSON array"):
        load_functions_definition(path)


def test_schema_mismatch_raises_input_file_error(tmp_path: Path) -> None:
    # Missing the required "returns" field.
    path = tmp_path / "bad_schema.json"
    path.write_text('[{"name": "fn_x", "description": "d", "parameters": {}}]', encoding="utf-8")
    with pytest.raises(InputFileError, match="schema"):
        load_functions_definition(path)


def test_wrong_prompt_field_type_raises_input_file_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_prompt.json"
    path.write_text('[{"prompt": 123}]', encoding="utf-8")  # prompt must be a string
    with pytest.raises(InputFileError):
        load_test_prompts(path)