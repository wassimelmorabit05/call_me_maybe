"""Keep generated tokens inside the JSON value grammar."""

import json
import re
from typing import Any

import numpy as np

from src.models import Parameter


def load_vocabulary(model: Any) -> dict:
    """Map Qwen vocabulary IDs to their original UTF-8 bytes."""
    with open(model.get_path_to_vocab_file(), encoding="utf-8") as file:
        vocabulary = json.load(file)
    visible = list(range(33, 127)) + list(range(161, 173))
    visible += list(range(174, 256))
    byte_map = {chr(value): value for value in visible}
    next_code = 256
    for value in range(256):
        if value not in visible:
            byte_map[chr(next_code)] = value
            next_code += 1
    tokens = {}
    for token, token_id in vocabulary.items():
        if not token.startswith("<|"):
            tokens[token_id] = bytes(byte_map[char] for char in token)
    return tokens


def string_prefix(text: str) -> bool:
    """Check an unfinished JSON string, including escape sequences."""
    if not text:
        return True
    if text[0] != '"':
        return False
    escaped = False
    hex_left = 0
    for char in text[1:]:
        if hex_left:
            if char not in "0123456789abcdefABCDEF":
                return False
            hex_left -= 1
        elif escaped:
            if char not in '"\\/bfnrtu':
                return False
            hex_left = 4 if char == "u" else 0
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"' or ord(char) < 32:
            return False
    return True


def number_prefix(text: str, kind: str) -> bool:
    """Check an unfinished integer or JSON number."""
    if text in ("", "-"):
        return True
    integer = r"-?(?:0|[1-9][0-9]*)"
    if kind == "integer":
        return re.fullmatch(integer, text) is not None
    base = integer + r"(?:\.[0-9]+)?"
    patterns = [base + r"(?:[eE][+-]?[0-9]+)?",
                base + r"[eE][+-]?", integer + r"\."]
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def value_state(text: str, schema: Parameter, boundary: str) -> int:
    """Return -1 for invalid, 0 for unfinished, or 1 for complete."""
    text = text.lstrip(" \t\r\n")
    if schema.enum is not None:
        for value in schema.enum:
            choice = json.dumps(value, ensure_ascii=False)
            if choice.startswith(text) and text != choice:
                return 0
            if text.startswith(choice):
                tail = text[len(choice):]
                if boundary.startswith(tail):
                    return int(bool(tail) or schema.type not in (
                        "integer", "number"
                    ))
        return -1
    if schema.type in ("number", "integer"):
        if number_prefix(text, schema.type):
            return 0
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        if schema.type == "string" and string_prefix(text):
            return 0
        choices = {"boolean": ["true", "false"], "null": ["null"]}
        if any(word.startswith(text) for word in choices.get(schema.type, [])):
            return 0
        return -1
    if not boundary.startswith(text[end:]):
        return -1
    try:
        schema.validate_value(value)
    except ValueError:
        return -1
    return 1


def token_state(data: bytes, schema: Parameter, boundary: str) -> int:
    """Validate a token prefix, allowing an unfinished UTF-8 character."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data":
            return -1
        if schema.enum is not None:
            choices = [json.dumps(value, ensure_ascii=False).encode("utf-8")
                       for value in schema.enum]
            prefix = data.lstrip(b" \t\r\n")
            if any(word.startswith(prefix) for word in choices):
                return 0
            return -1
        text = data[:error.start].decode("utf-8")
        text = text.lstrip(" \t\r\n")
        if schema.type != "string" or not text.startswith('"'):
            return -1
        return 0 if string_prefix(text + "x") else -1
    return value_state(text, schema, boundary)


def generate_value(context: str, schema: Parameter, boundary: str,
                   model: Any, tokens: dict) -> Any:
    """Generate one typed value by masking invalid tokens before selection."""
    input_ids = model.encode(context).tolist()[0]
    generated = b""
    for _ in range(256):
        logits = np.asarray(model.get_logits_from_input_ids(input_ids))
        logits[~np.isfinite(logits)] = -np.inf
        while True:
            token_id = int(np.argmax(logits))
            if not np.isfinite(logits[token_id]):
                raise ValueError("The model has no valid next token")
            addition = tokens.get(token_id, b"")
            state = token_state(generated + addition, schema, boundary)
            if addition and state >= 0:
                break
            logits[token_id] = -np.inf
        generated += addition
        input_ids.append(token_id)
        if state == 1:
            value = json.JSONDecoder().raw_decode(
                generated.decode("utf-8").lstrip()
            )[0]
            return schema.validate_value(value)
    raise ValueError("A value exceeded 256 tokens; no output was written")
