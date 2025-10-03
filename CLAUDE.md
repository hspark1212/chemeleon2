# chemeleon2 Development Guidelines

Auto-generated from all feature plans. Last updated: 2025-10-03

## Active Technologies
- Python 3.11+ + Ruff (formatter + linter), pre-commit framework, GitHub Actions (001-this-repository-was)
- Python 3.11+ (already specified in pyproject.toml) + Ruff (formatter + linter), pre-commit framework, pytest, GitHub Actions (001-this-repository-was)
- Configuration files (.toml, .yaml), git repository (001-this-repository-was)

## Development Environment
- **Python**: Use `.venv/bin/python` (Python 3.11.13 in virtual environment)
- **Virtual Environment**: `.venv/` at repository root (already activated in shell)
- **Package Manager**: uv

## Project Structure
```
src/
tests/
.venv/        # Virtual environment - ALWAYS use this Python
```

## Commands
cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style
Python 3.11+: Follow standard conventions

## Recent Changes
- 001-this-repository-was: Added Python 3.11+ (already specified in pyproject.toml) + Ruff (formatter + linter), pre-commit framework, pytest, GitHub Actions
- 001-this-repository-was: Added Python 3.11+ + Ruff (formatter + linter), pre-commit framework, GitHub Actions

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
