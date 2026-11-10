set shell := ["bash", "-cu"]

# List available recipes
default:
    @just --list

# Create .venv and install dependencies
venv:
    uv venv
    uv sync
