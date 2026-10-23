from src.decoder import can_end_number, is_number_prefix


def test_number_prefix() -> None:
    assert is_number_prefix("-12.5") is True


def test_complete_number() -> None:
    assert can_end_number("42") is True
