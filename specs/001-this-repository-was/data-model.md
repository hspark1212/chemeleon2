# Data Model: Development Workflow Standards

**Feature**: 001-this-repository-was
**Date**: 2025-10-03
**Status**: Design Phase

## Overview

This feature is configuration-driven and does not involve runtime data models or databases. Instead, it defines **configuration entities** that represent structured data in configuration files.

---

## Entity 1: Ruff Configuration

**Location**: `pyproject.toml` → `[tool.ruff]` section

**Purpose**: Defines formatting and linting rules for Python code

**Schema**:
```toml
[tool.ruff]
# Core settings
target-version = "py311"              # Python version target
line-length = 88                      # Black-compatible line length

# Linting rules
select = [                            # Rule sets to enable
    "F",    # Pyflakes
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "I",    # isort
    "N",    # pep8-naming
    "D",    # pydocstyle
    "UP",   # pyupgrade
    "ANN",  # flake8-annotations
    "S",    # flake8-bandit
    "B",    # flake8-bugbear
    "C90",  # mccabe complexity
]

# Validation rules
[tool.ruff.lint.pydocstyle]
convention = "google"                 # Docstring style

[tool.ruff.lint.mccabe]
max-complexity = 10                   # Complexity threshold
```

**Attributes**:
- `target-version`: String (enum: py311, py312, etc.)
- `line-length`: Integer (88 for Black compatibility)
- `select`: List of rule code strings
- `convention`: String (google, numpy, pep257)
- `max-complexity`: Integer (complexity limit)

**Validation Rules**:
- `target-version` must match project's `requires-python` in pyproject.toml
- `line-length` must be 88 (Black default, per clarifications)
- `select` must include minimum required rule sets (F, E, W, I, N, D, UP, ANN, S, B, C90)
- `max-complexity` should be <= 10 (strict quality standard)

**Relationships**:
- Referenced by pre-commit hooks (Ruff uses this config)
- Referenced by CI/CD workflow (same config for consistency)

---

## Entity 1a: Pyright Configuration

**Location**: `pyproject.toml` → `[tool.pyright]` section

**Purpose**: Defines type checking mode for Python code

**Schema**:
```toml
[tool.pyright]
# Type checking mode
typeCheckingMode = "basic"            # basic, standard, or strict
# Python version target
pythonVersion = "3.11"                # Python version
```

**Attributes**:
- `typeCheckingMode`: String (enum: basic, standard, strict)
- `pythonVersion`: String (e.g., "3.11", "3.12")

**Validation Rules**:
- `typeCheckingMode` must be "basic" (realistic for current codebase)
- `pythonVersion` must match project's `requires-python` in pyproject.toml

**Relationships**:
- Referenced by pre-commit hooks (pyright uses this config)
- Referenced by CI/CD workflow (same config for consistency)

---

## Entity 2: Pre-commit Configuration

**Location**: `.pre-commit-config.yaml` at repository root

**Purpose**: Defines git hooks that run before commits

**Schema**:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.x.x                        # Ruff version (pinned)
    hooks:
      - id: ruff-format                # Formatter hook
        name: "Ruff Format"
        stages: [commit]
      - id: ruff
        name: "Ruff Lint"
        args: [--fix]                  # Auto-fix when possible
        stages: [commit]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.x.x
    hooks:
      - id: check-yaml                 # YAML validation
        files: \.yaml$
      - id: check-toml                 # TOML validation
        files: \.toml$
      - id: check-json                 # JSON validation
        files: \.json$
```

**Attributes**:
- `repos`: List of repository configurations
  - `repo`: String (Git repository URL)
  - `rev`: String (version/tag, pinned for consistency)
  - `hooks`: List of hook configurations
    - `id`: String (hook identifier)
    - `name`: String (display name)
    - `args`: List of strings (CLI arguments)
    - `stages`: List of strings (when to run: commit, push, etc.)
    - `files`: String (regex pattern for file matching)

**Validation Rules**:
- `rev` must be pinned version (not branch or "latest")
- Ruff hooks must run before validation hooks (order matters)
- `ruff-format` must run before `ruff` lint (formatting first)
- File-specific hooks must have `files` regex pattern

**State Transitions**:
1. **Not installed** → Developer clones repo
2. **Installed** → Setup script runs `pre-commit install`
3. **Active** → Hooks run on every `git commit`
4. **Updated** → `pre-commit autoupdate` updates hook versions

**Relationships**:
- Depends on Ruff Configuration (uses `pyproject.toml` settings)
- Triggers validation on files matching `files` patterns

---

## Entity 3: GitHub Actions Workflow

**Location**: `.github/workflows/ci.yml`

**Purpose**: CI/CD pipeline for Pull Request validation

**Schema**:
```yaml
name: CI

on:
  pull_request:                        # Trigger on PRs
    branches: [main, develop]
  push:
    branches: [main]                   # Trigger on main pushes

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install ruff pytest
          pip install -e .               # Install project

      - name: Ruff format check
        run: ruff format --check .

      - name: Ruff lint check
        run: ruff check .

      - name: Run tests
        run: pytest
```

**Attributes**:
- `name`: String (workflow display name)
- `on`: Object (trigger configuration)
  - `pull_request.branches`: List of branch names
  - `push.branches`: List of branch names
- `jobs`: Object (job definitions)
  - Job name → Job configuration
    - `runs-on`: String (runner environment)
    - `steps`: List of step configurations
      - `uses`: String (action to use) OR
      - `name` + `run`: String (shell command)

**Validation Rules**:
- Must run on `pull_request` events
- Python version must match `target-version` in Ruff config (3.11+)
- Ruff commands must match pre-commit hook behavior
- Tests must run after linting (fail-fast on lint errors)

**State Transitions**:
1. **PR opened** → Workflow triggered
2. **Running** → Executing steps sequentially
3. **Success** → All steps passed, PR can merge
4. **Failure** → At least one step failed, PR blocked

**Relationships**:
- Uses Ruff Configuration (same rules as pre-commit)
- Blocks PR merge if validation fails (required status check)

---

## Entity 4: Setup Script Interface

**Location**: Repository root (e.g., `setup-dev.sh` or `Makefile` with `setup` target)

**Purpose**: One-command developer environment setup

**Interface Contract**:
```bash
# Command signature
$ ./setup-dev.sh [options]

# or via Make
$ make setup

# Expected behavior
1. Check Python version >= 3.11
2. Install pre-commit framework
3. Install Ruff
4. Run `pre-commit install`
5. Optionally: Run `pre-commit run --all-files` for validation

# Exit codes
0 = Success
1 = Python version too old
2 = Installation failed
3 = Pre-commit install failed
```

**Attributes**:
- Input: None (or optional flags like `--skip-validation`)
- Output: Success/failure messages to stdout/stderr
- Side effects: Installs tools, creates `.git/hooks/` files

**Validation Rules**:
- Must be idempotent (safe to run multiple times)
- Must check prerequisites before installing
- Must provide clear error messages

**Relationships**:
- Creates Pre-commit Configuration hooks in `.git/hooks/`
- Documented in CONTRIBUTING.md

---

## Entity 5: Developer Documentation

**Location**: `CONTRIBUTING.md` at repository root

**Purpose**: Team onboarding and workflow documentation

**Structure**:
```markdown
# Contributing to chemeleon2

## Table of Contents
1. Setup Instructions
2. Development Workflow
3. Coding Standards
4. AI Agent Usage
5. Troubleshooting
6. CI/CD

## 1. Setup Instructions
[One-command setup, prerequisites]

## 2. Development Workflow
[Commit process, pre-commit expectations]

## 3. Coding Standards
[Ruff rules, line length, type hints policy]

## 4. AI Agent Usage
[How to ask Claude/Codex/Gemini to fix violations]
[Example prompts]

## 5. Troubleshooting
[Common issues and solutions]

## 6. CI/CD
[GitHub Actions workflow, required checks]
```

**Attributes**:
- Sections: List of documentation sections
- Examples: Code snippets, command examples
- Links: References to tools, external docs

**Validation Rules**:
- Must cover all setup steps from setup script
- Must provide AI agent usage examples (per FR-015)
- Must explain Ruff rule choices (per FR-008)

**Relationships**:
- References Setup Script (installation instructions)
- References Ruff Configuration (coding standards)
- References CI/CD Workflow (PR process)

---

## Entity 6: Ruff Execution Result

**Runtime Entity** (not persisted, exists during validation)

**Purpose**: Represents output from Ruff formatter/linter

**Schema** (JSON output with `--output-format=json`):
```json
{
  "files": [
    {
      "path": "src/models/example.py",
      "violations": [
        {
          "code": "D103",
          "message": "Missing docstring in public function",
          "location": {
            "row": 42,
            "column": 1
          },
          "fix": {
            "content": "\"\"\"Brief description.\"\"\"\n",
            "location": {
              "row": 42,
              "column": 1
            }
          }
        }
      ]
    }
  ]
}
```

**Attributes**:
- `files`: List of file results
  - `path`: String (relative file path)
  - `violations`: List of violation objects
    - `code`: String (Ruff rule code, e.g., "D103")
    - `message`: String (human-readable description)
    - `location`: Object (row, column)
    - `fix`: Optional object (auto-fix suggestion)

**State Transitions**:
1. **Executing** → Ruff scanning files
2. **Complete** → Results available
3. **Consumed** → Developer or AI agent reads results

**Relationships**:
- Generated by Ruff based on Ruff Configuration
- Consumed by developer or AI agent for fixing
- Reported to CI/CD as pass/fail status

---

## Configuration File Summary

| Entity | File Location | Format | Owner |
|--------|--------------|--------|-------|
| Ruff Configuration | `pyproject.toml` | TOML | Repository |
| Pre-commit Configuration | `.pre-commit-config.yaml` | YAML | Repository |
| CI/CD Workflow | `.github/workflows/ci.yml` | YAML | Repository |
| Setup Script | `setup-dev.sh` or `Makefile` | Shell/Make | Repository |
| Developer Documentation | `CONTRIBUTING.md` | Markdown | Repository |

---

## Validation Matrix

| Entity | Validated By | When | Failure Impact |
|--------|-------------|------|----------------|
| Ruff Configuration | Ruff CLI | Pre-commit, CI | Commit blocked or PR blocked |
| Pre-commit Configuration | Pre-commit framework | On install | Setup fails |
| CI/CD Workflow | GitHub Actions | PR open/update | PR validation fails |
| Setup Script | Manual testing | Development | Setup fails |
| Developer Documentation | Manual review | PR review | Documentation incomplete |

---

## Data Flow

```
1. Developer modifies code
   ↓
2. Runs `git commit`
   ↓
3. Pre-commit hooks trigger
   ↓
4. Ruff reads pyproject.toml
   ↓
5. Ruff executes format check + lint
   ↓
6. Violations found?
   → YES: Commit blocked, show Ruff Execution Result
   → NO: Commit proceeds
   ↓
7. Push to GitHub
   ↓
8. Open PR
   ↓
9. GitHub Actions CI triggered
   ↓
10. CI runs same Ruff checks + pytest
    ↓
11. All checks pass?
    → YES: PR can merge
    → NO: PR blocked, developer fixes
```

---

## Contract Testing Focus

Contract tests will validate:
1. **Ruff Configuration**: Exists, is valid TOML, has required fields
2. **Pre-commit Configuration**: Exists, is valid YAML, has Ruff hooks
3. **CI Workflow**: Exists, is valid YAML, runs Ruff + pytest
4. **Setup Script**: Exists, is executable, runs without errors

See `contracts/` directory for formal schemas.

---

**Data Model Complete** ✅
All configuration entities defined. Ready for contract generation.
