# ABOUTME: File loading/saving helpers with graceful JSON and schema error handling.
# ABOUTME: Never lets a malformed or missing input file crash the program.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn

from pydantic import ValidationError

from .models import FunctionCallResult, FunctionsDefinition, PromptsFile


def _fail(message: str) -> NoReturn:
    print(f"[error] {message}", file=sys.stderr)
    sys.exit(1)


def load_json(path: Path) -> Any:
    """Load raw JSON from `path`, exiting with a clear message on any error."""
    if not path.exists():
        _fail(f"input file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON in {path}: {exc}")
    except OSError as exc:
        _fail(f"could not read {path}: {exc}")


def load_functions_definition(path: Path) -> FunctionsDefinition:
    raw = load_json(path)
    try:
        return FunctionsDefinition.model_validate(raw)
    except ValidationError as exc:
        _fail(f"functions definition file does not match the expected schema:\n{exc}")


def load_prompts(path: Path) -> PromptsFile:
    raw = load_json(path)
    try:
        return PromptsFile.model_validate(raw)
    except ValidationError as exc:
        _fail(f"test prompts file does not match the expected schema:\n{exc}")


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump() for r in results]
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError as exc:
        _fail(f"could not write output file {path}: {exc}")
