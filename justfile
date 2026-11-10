set shell := ["bash", "-cu"]

# List available recipes
default:
    @just --list

# Create .venv and install dependencies
venv:
    uv venv
    uv sync

# Run the test suite
test:
    uv run pytest

# Lint the codebase
lint:
    uv run ruff check .
