# Contract: Setup Script

**File**: `setup-dev.sh` or `Makefile` with `setup` target
**Purpose**: One-command developer environment setup
**Version**: 1.0.0
**Date**: 2025-10-03

## Interface

```bash
# Option 1: Shell script
$ ./setup-dev.sh

# Option 2: Makefile
$ make setup
```

## Required Behavior

1. Check Python version >= 3.11
2. Install pre-commit framework
3. Install Ruff
4. Run `pre-commit install`
5. Print success message

## Exit Codes

- 0: Success
- 1: Python version too old
- 2: Installation failed

## Validation Rules

1. Script must exist and be executable
2. Must check Python version
3. Must install required tools
4. Must be idempotent (safe to run multiple times)
5. Must provide clear error messages

## Test Location

`tests/contract/test_setup_script.py`
