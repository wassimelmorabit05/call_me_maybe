.PHONY: install run debug bonus lint clean



install:
	uv sync

run:
	uv run python -m src 

debug:
	uv run python -m pdb -m src 

bonus:
	uv run python -m src --visualize 

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	find src -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache
