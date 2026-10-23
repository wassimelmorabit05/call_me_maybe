import json


def load_vocabulary(
    vocab_path: str,
) -> dict[str, int]:
    """Load the model vocabulary."""

    with open(
        vocab_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Vocabulary must be a JSON object."
        )

    vocabulary: dict[str, int] = {}

    for token, token_id in data.items():
        if not isinstance(token, str):
            raise ValueError(
                "Vocabulary tokens must be strings."
            )

        if isinstance(token_id, bool) or not isinstance(
            token_id,
            int,
        ):
            raise ValueError(
                "Vocabulary token IDs must be integers."
            )

        vocabulary[token] = token_id

    return vocabulary
