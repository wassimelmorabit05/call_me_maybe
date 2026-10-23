import json
from pathlib import Path

from .models import FunctionCallResult


def write_results(
    path: str,
    results: list[FunctionCallResult],
) -> None:
    """Write function call results to a JSON file."""

    data = [
        result.model_dump()
        for result in results
    ]

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )
