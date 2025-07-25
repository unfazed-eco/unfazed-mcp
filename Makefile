all: test

test:
	@echo "Running tests..."
	uv run pytest -v -s --cov ./unfazed_mcp --cov-report term-missing

format:
	@echo "Formatting code..."
	uv run ruff format tests/ unfazed_mcp/
	uv run ruff check tests/ unfazed_mcp/  --fix
	uv run mypy --check-untyped-defs --explicit-package-bases --ignore-missing-imports tests/ unfazed_mcp/

publish:
	@echo "Publishing package..."
	uv build
	uv publish
