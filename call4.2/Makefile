
.PHONY: install run debug lint lint-strict test bonus clean

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 src tests
	uv run mypy src tests \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src tests
	uv run mypy src tests --strict --ignore-missing-imports

test:
	uv run --with pytest pytest -q

bonus:
	uv run python -m src --visualize

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
