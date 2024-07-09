#!/bin/bash

# Create and activate a new Conda environment
conda create -n symbac_development python=3.12 -y
conda activate symbac_development

# Install the package in editable mode with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Generate requirements files
pip-compile pyproject.toml -o requirements.txt
pip-compile pyproject.toml --extra dev -o requirements-dev.txt

echo "Development environment setup complete!"
echo "To activate the environment, run: conda activate symbac_dev"
