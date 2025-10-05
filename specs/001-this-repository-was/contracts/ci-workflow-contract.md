# Contract: GitHub Actions CI Workflow

**File**: `.github/workflows/ci.yml`
**Purpose**: CI/CD pipeline for PR validation
**Version**: 1.0.0
**Date**: 2025-10-03

## Required Structure

```yaml
name: CI

on:
  pull_request:
    branches: [main, develop]  # Or your branch names
  push:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff pytest
      - run: ruff format --check .
      - run: ruff check .
      - run: pytest
```

## Validation Rules

1. File must exist at `.github/workflows/ci.yml`
2. Must be valid YAML syntax
3. Must trigger on pull_request events
4. Must use Python 3.11+
5. Must run ruff format check
6. Must run ruff lint check
7. Must run pytest
8. Steps must run in order (lint before tests)

## Test Location

`tests/contract/test_github_actions.py`
