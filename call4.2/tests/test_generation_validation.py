"""Regression tests for JSON-compatible constrained value generation."""

import pytest

from src.call_generator import (
    _is_complete_number,
    _is_number_prefix,
    _is_valid_string_addition,
    _pending_escape,
    _unescape_string,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", True),
        ("-", True),
        ("0", True),
        ("-0", True),
        ("1", True),
        ("01", False),
        ("-01", False),
        ("1.", True),
        ("1.2", True),
        ("1e", True),
        ("1e+", True),
        ("1e3", True),
        ("1.e2", False),
        ("-2.5e-4", True),
        ("١٢", False),
        ("+2", False),
        ("1..2", False),
    ],
)
def test_json_number_prefix(value: str, expected: bool) -> None:
    assert _is_number_prefix(value, allow_dot=True) is expected


def test_integer_rejects_decimal_and_exponent() -> None:
    assert _is_number_prefix("12", allow_dot=False)
    assert not _is_number_prefix("12.0", allow_dot=False)
    assert not _is_number_prefix("1e3", allow_dot=False)
    assert _is_complete_number("12", allow_dot=False)
    assert not _is_complete_number("12.0", allow_dot=False)


def test_complete_json_number_rules() -> None:
    assert _is_complete_number("1e3")
    assert _is_complete_number("-2.5e-4")
    assert not _is_complete_number("01")
    assert not _is_complete_number("1.")
    assert not _is_complete_number("1e+")
    assert not _is_number_prefix("1.e", allow_dot=True)


def test_string_rejects_unescaped_control_characters() -> None:
    assert not _is_valid_string_addition("", "hello\nworld")
    assert not _is_valid_string_addition("", "hello\tworld")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (r"a\nb", "a\nb"),
        (r"a\tb", "a\tb"),
        (r"\u0041", "A"),
        (r"\"", '"'),
        (r"\\", "\\"),
        (r"\/", "/"),
    ],
)
def test_string_supports_json_escapes(raw: str, expected: str) -> None:
    assert _is_valid_string_addition("", raw)
    assert _unescape_string(raw) == expected


def test_incomplete_unicode_escape_is_not_a_string_boundary() -> None:
    assert _pending_escape("prefix\\u12")
    assert not _pending_escape("prefix\\u1234")


def test_function_definition_rejects_invalid_parameter_names() -> None:
    from pydantic import ValidationError

    from src.input_loader import FunctionDefinition

    definition = {
        "name": "fn_example",
        "description": "Example function",
        "parameters": {"bad-name": {"type": "string"}},
        "returns": {"type": "string"},
    }
    with pytest.raises(ValidationError, match="Invalid parameter name"):
        FunctionDefinition.model_validate(definition)
