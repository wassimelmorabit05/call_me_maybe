.PHONY: install run debug clean lint test

install:
	@uv sync

run:
	@uv run python -m src

debug:
	@uv run python -m pdb -m src

clean:
	@find src tests llm_sdk -type f -name "*.pyc" -delete
	@find src tests llm_sdk -type d -name "__pycache__" -prune -exec rm -rf {} +
	@rm -rf .mypy_cache .pytest_cache

lint:
	@uv run flake8 src tests
	@uv run mypy src tests --warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

test:
	@uv run pytest -v