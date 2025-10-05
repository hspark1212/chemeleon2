# Contract: Pre-commit Configuration

**File**: `.pre-commit-config.yaml`
**Purpose**: Define git hooks for pre-commit validation
**Version**: 1.0.0
**Date**: 2025-10-03

## Required Structure

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.x.x  # Pinned version (not branch or latest)
    hooks:
      - id: ruff-format
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.x.x
    hooks:
      - id: check-yaml
      - id: check-toml
      - id: check-json
```

## Validation Rules

1. File must exist at repository root
2. Must be valid YAML syntax
3. Must include ruff-pre-commit repo
4. Must include ruff-format and ruff hooks
5. Must include YAML, TOML, JSON validators
6. Versions must be pinned (not "latest")
7. Must NOT include pytest or test execution hooks

## Test Location

`tests/contract/test_precommit_config.py`
