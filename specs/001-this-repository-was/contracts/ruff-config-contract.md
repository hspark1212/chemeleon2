# Contract: Ruff Configuration

**File**: `pyproject.toml` → `[tool.ruff]` section
**Purpose**: Define formatting and linting rules for Python code
**Version**: 1.0.0
**Date**: 2025-10-03

## Required Sections

### [tool.ruff]
```toml
[tool.ruff]
target-version = "py311"      # MUST be py311 or higher
line-length = 88              # MUST be 88 (Black default)
```

### [tool.ruff.lint]
```toml
[tool.ruff.lint]
select = [
    "F", "E", "W", "I", "N", "D", "UP", "ANN", "S", "B", "C90"
]
```

## Validation Rules

1. File must exist at repository root
2. Must be valid TOML syntax
3. target-version >= py311
4. line-length == 88
5. All 11 rule sets must be in select array

## Test Location

`tests/contract/test_ruff_config.py`
