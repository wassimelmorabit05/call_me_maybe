import json
from pathlib import Path

import pytest

from src.loader import load_json


def test_load_json(tmp_path: Path) -> None:
    file_path = tmp_path / "test.json"

    data = {
        "name": "test",
        "value": 42,
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file)

    result = load_json(str(file_path))

    assert result == data


def test_load_invalid_json(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.json"

    file_path.write_text(
        '{"name": "test",}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_json(str(file_path))
