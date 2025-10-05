# Contributing to chemeleon2

Welcome to the chemeleon2 project! This guide will help you set up your development environment and understand our development workflow standards.

## Table of Contents

1. [Setup Instructions](#1-setup-instructions)
2. [Development Workflow](#2-development-workflow)
3. [Coding Standards](#3-coding-standards)
4. [Type Checking](#4-type-checking)
5. [Troubleshooting](#5-troubleshooting)
6. [CI/CD](#6-cicd)
7. [AI Agent Usage](#7-ai-agent-usage)

---

## 1. Setup Instructions

### Prerequisites

- **Python 3.11+** installed on your system
- **Git** installed
- **uv** package manager (recommended for virtual environment management)

If you don't have `uv` installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### One-Command Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/YOUR_ORG/chemeleon2.git
   cd chemeleon2
   ```

2. **Create virtual environment** (if not already created):
   ```bash
   uv venv
   ```

3. **Run the setup script**:
   ```bash
   ./setup-dev.sh
   ```

**Expected output**:
```
=========================================
Development Environment Setup
=========================================

Checking virtual environment... ✓ Found
Checking Python version... ✓ Python 3.11.13
Checking uv package manager... ✓ Found
Installing pre-commit... ✓ Done
Installing Ruff... ✓ Done
Installing pyright... ✓ Done
Installing pre-commit hooks... ✓ Done

✓ Development environment setup complete!
```

### What the Setup Script Does

The `setup-dev.sh` script automatically:
1. Verifies Python 3.11+ is installed in `.venv/`
2. Verifies `uv` package manager is available
3. Installs the `pre-commit` framework
4. Installs `Ruff` (formatter and linter)
5. Installs `pyright` (type checker)
6. Installs pre-commit hooks into your `.git/hooks/` directory
7. Runs pre-commit validation on all files

### Verify Installation

Run pre-commit on all files to verify everything is working:

```bash
.venv/bin/pre-commit run --all-files
```

**Expected output**:
```
Ruff Format.....................................Passed
Ruff Lint.......................................Passed
Pyright Type Check..............................Passed
Check YAML......................................Passed
Check TOML......................................Passed
Check JSON......................................Passed
```

---

## 2. Development Workflow

### Before Making Changes: Run Baseline Tests

**CRITICAL FIRST STEP**: Before applying any changes, verify the code works correctly by running baseline tests.

```bash
pytest tests/baseline/ -v
```

**Expected output**:
```
tests/baseline/test_vae_module.py::test_vae_instantiation PASSED
tests/baseline/test_ldm_module.py::test_ldm_instantiation PASSED
tests/baseline/test_rl_module.py::test_rl_module_instantiation PASSED
tests/baseline/test_data_loading.py::test_dataloader_batching PASSED
==================== 4 passed in 2.35s ====================
```

**If tests fail**: Fix the code issues BEFORE proceeding. Baseline tests ensure that functionality is preserved during development.

### Making Code Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** to the code

3. **Stage your changes**:
   ```bash
   git add .
   ```

4. **Commit your changes**:
   ```bash
   git commit -m "feat: your feature description"
   ```

### What Happens When You Commit

When you run `git commit`, pre-commit hooks automatically run:

1. **Ruff Format Check** - Ensures code is properly formatted
2. **Ruff Lint Check** - Checks for code quality issues
3. **Pyright Type Check** - Validates type annotations
4. **YAML/TOML/JSON Validation** - Checks configuration files

**If all checks pass**: Your commit succeeds.

**If checks fail**: The commit is blocked and you'll see error messages explaining what needs to be fixed.

### Example: Commit Blocked by Violations

```bash
$ git commit -m "test: add new feature"

Ruff Format.....................................Failed
- hook id: ruff-format
- files were modified by this hook

src/models/new_feature.py

Ruff Lint.......................................Failed
- hook id: ruff
src/models/new_feature.py:10:5: F841 Local variable `x` is assigned to but never used
src/models/new_feature.py:15:1: D103 Missing docstring in public function
src/models/new_feature.py:20:15: E225 Missing whitespace around operator
```

You have three options to fix these violations:

1. **Fix manually** and re-commit
2. **Ask an AI agent** to fix (see [Section 7: AI Agent Usage](#7-ai-agent-usage))
3. **Skip hooks** with `--no-verify` (NOT RECOMMENDED - CI will still catch violations)

### After Fixing Violations

```bash
# Stage your fixes
git add .

# Try committing again
git commit -m "test: add new feature"
```

**Expected output when successful**:
```
Ruff Format.....................................Passed
Ruff Lint.......................................Passed
Pyright Type Check..............................Passed
[feature/your-feature abc1234] test: add new feature
 1 file changed, 10 insertions(+)
```

### Pushing and Creating Pull Requests

```bash
# Push your branch to GitHub
git push origin feature/your-feature-name

# Open a Pull Request on GitHub
# CI checks will run automatically
```

---

## 3. Coding Standards

### Formatter: Ruff Format

We use **Ruff Format** for code formatting, configured to be **Black-compatible**.

**Key settings** (from `pyproject.toml`):
- **Line length**: 88 characters (Black default)
- **Target Python version**: 3.11+
- **Style**: Black-compatible opinionated formatting

**Common formatting rules**:
- Double quotes for strings (not single quotes)
- Proper spacing around operators (`x = 1 + 2`, not `x=1+2`)
- Consistent indentation (4 spaces)
- Trailing commas in multi-line structures

### Linter: Ruff

We enforce **strict linting rules** to maintain code quality.

**Enabled rule sets**:

| Code | Rule Set | Purpose |
|------|----------|---------|
| **F** | Pyflakes | Catches undefined names, unused imports |
| **E/W** | pycodestyle | PEP 8 style errors and warnings |
| **I** | isort | Import statement sorting |
| **N** | pep8-naming | Naming conventions |
| **D** | pydocstyle | Docstring requirements (Google style) |
| **UP** | pyupgrade | Modern Python syntax |
| **ANN** | flake8-annotations | Type hint coverage |
| **S** | flake8-bandit | Security checks |
| **B** | flake8-bugbear | Design issues and anti-patterns |
| **C90** | mccabe | Cyclomatic complexity (max 10) |

**Why these rules?**

- **F, E, W**: Essential Python style and correctness
- **I, N**: Consistency in imports and naming
- **D**: Documentation is required for team collaboration
- **UP**: Leverage modern Python 3.11+ features
- **ANN**: Type hints improve code clarity and catch bugs
- **S**: Prevent common security vulnerabilities
- **B**: Catch subtle bugs and anti-patterns
- **C90**: Enforce low complexity for maintainability

**Note**: Some rules are temporarily ignored for the existing codebase (see `pyproject.toml` → `[tool.ruff.lint.ignore]`). New code should follow all standards.

### Docstring Style

We use **Google-style docstrings**:

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief one-line description.

    Longer description if needed, explaining the function's purpose
    and behavior in more detail.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    return len(param2) > param1
```

### Complexity Limit

Functions must have a **cyclomatic complexity ≤ 10**. If a function is too complex, break it into smaller functions.

---

## 4. Type Checking

### Pyright: Basic Mode

We use **pyright** in **basic mode** for type checking.

**Configuration** (from `pyproject.toml`):
```toml
[tool.pyright]
typeCheckingMode = "basic"
pythonVersion = "3.11"
```

**Why basic mode?**
- Provides essential type checking without overwhelming strictness
- Practical for ML/scientific code with complex data structures
- Allows gradual type hint adoption
- Catches common type errors while remaining developer-friendly

### Type Hint Best Practices

```python
# Good: Clear type hints
def process_data(data: list[float], threshold: float = 0.5) -> list[float]:
    """Process data above threshold."""
    return [x for x in data if x > threshold]

# Good: Complex types
from typing import Optional, Union
from pathlib import Path

def load_model(checkpoint: Union[str, Path]) -> Optional[torch.nn.Module]:
    """Load model from checkpoint path."""
    if not Path(checkpoint).exists():
        return None
    return torch.load(checkpoint)

# Good: Type aliases for clarity
from typing import TypeAlias

TensorBatch: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

def collate_batch(items: list[dict]) -> TensorBatch:
    """Collate items into tensor batch."""
    # Implementation
    pass
```

### When Type Checking Fails

Pyright errors will block your commit. Common fixes:

1. **Add missing type hints**:
   ```python
   # Before (fails)
   def process(data):
       return data * 2

   # After (passes)
   def process(data: float) -> float:
       return data * 2
   ```

2. **Use `Optional` for values that can be None**:
   ```python
   from typing import Optional

   def find_item(items: list[str], target: str) -> Optional[str]:
       return target if target in items else None
   ```

3. **Use `Any` as an escape hatch** (sparingly):
   ```python
   from typing import Any

   def complex_function(data: Any) -> Any:
       # When types are too complex to specify
       pass
   ```

---

## 5. Troubleshooting

### Common Setup Issues

#### "Python version too old"

**Error**:
```
ERROR: Python 3.11+ required.
```

**Solution**: Install Python 3.11 or higher:
```bash
python --version  # Check your version
# Install Python 3.11+ via your system package manager
```

#### "uv: command not found"

**Error**:
```
ERROR: uv package manager is required.
```

**Solution**: Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart your shell or source your profile
```

#### "pre-commit: command not found"

**Solution**: Re-run setup script or install manually:
```bash
./setup-dev.sh
# OR
uv pip install pre-commit
```

#### ".venv/bin/python not found"

**Solution**: Create virtual environment first:
```bash
uv venv
./setup-dev.sh
```

### Common Pre-commit Issues

#### Hooks not running on commit

**Solution**: Reinstall hooks:
```bash
.venv/bin/pre-commit install
```

#### "Hook failed to run" errors

**Solution**: Update pre-commit and hooks:
```bash
.venv/bin/pre-commit autoupdate
.venv/bin/pre-commit install --install-hooks
```

#### Ruff format check fails

**Error**:
```
Ruff Format.....................................Failed
- files were modified by this hook
```

**Solution**: The formatter automatically fixed your files. Just re-stage and commit:
```bash
git add .
git commit -m "your message"
```

#### Ruff lint violations

**Error**:
```
src/file.py:10:5: F841 Local variable `x` is assigned to but never used
```

**Solution options**:
1. Fix manually (remove unused variable)
2. Use Ruff's auto-fix: `.venv/bin/ruff check --fix .`
3. Ask an AI agent to fix (see [Section 7](#7-ai-agent-usage))

#### Pyright type errors

**Error**:
```
src/file.py:15:10 - error: Argument of type "str" cannot be assigned to parameter "x" of type "int"
```

**Solution**: Add or fix type annotations:
```python
# Before (fails)
def process(x):
    return x * 2

# After (passes)
def process(x: int) -> int:
    return x * 2
```

### Common Type Checking Errors

#### Missing return type annotation

**Error**:
```
error: Return type annotation missing
```

**Solution**:
```python
# Add return type annotation
def calculate() -> float:
    return 42.0
```

#### Type mismatch

**Error**:
```
error: Type "str" cannot be assigned to type "int"
```

**Solution**: Ensure types match or use Union:
```python
from typing import Union

def flexible(value: Union[str, int]) -> str:
    return str(value)
```

---

## 6. CI/CD

### GitHub Actions Workflow

Our CI pipeline runs automatically on:
- **Pull requests** to `main` or `develop` branches
- **Pushes** to `main` branch

**CI Steps**:
1. **Checkout code** - Retrieves your PR code
2. **Set up Python 3.11** - Configures Python environment
3. **Install dependencies** - Installs project + dev dependencies
4. **Ruff format check** - Verifies formatting (same as pre-commit)
5. **Ruff lint check** - Verifies code quality (same as pre-commit)
6. **Pyright type check** - Validates type annotations (same as pre-commit)
7. **Run tests** - Executes full test suite with pytest

**All checks must pass** before your PR can be merged.

### Viewing CI Results

1. Open your Pull Request on GitHub
2. Scroll to the "Checks" section at the bottom
3. Click "Details" on any failed check to see logs

### CI/CD Failure Troubleshooting

#### CI fails but local pre-commit passes

**Cause**: Version mismatch between local and CI

**Solution**: Update pre-commit hooks to match CI:
```bash
.venv/bin/pre-commit autoupdate
.venv/bin/pre-commit run --all-files
```

#### "Ruff format check" fails in CI

**Error in CI logs**:
```
Error: Process completed with exit code 1.
ruff format --check . failed
```

**Solution**: Run format check locally and commit fixes:
```bash
.venv/bin/ruff format --check .
# If files need formatting:
.venv/bin/ruff format .
git add .
git commit -m "fix: apply ruff formatting"
git push
```

#### "Ruff lint check" fails in CI

**Error in CI logs**:
```
src/models/example.py:42:5: F841 Local variable `temp` is assigned to but never used
```

**Solution**: Fix the violation locally:
```bash
.venv/bin/ruff check .
# Auto-fix if possible:
.venv/bin/ruff check --fix .
git add .
git commit -m "fix: resolve ruff lint violations"
git push
```

#### "Pyright type check" fails in CI

**Error in CI logs**:
```
error: Type annotation missing for parameter "data"
```

**Solution**: Add type annotations:
```python
# Fix the function signature
def process_data(data: torch.Tensor) -> torch.Tensor:
    return data * 2
```

Then commit and push:
```bash
git add .
git commit -m "fix: add type annotations"
git push
```

#### "Run tests" fails in CI

**Error in CI logs**:
```
FAILED tests/test_module.py::test_function - AssertionError
```

**Solution**: Run tests locally to debug:
```bash
pytest tests/ -v
# Or run specific test:
pytest tests/test_module.py::test_function -v
```

Fix the test or code, then push:
```bash
git add .
git commit -m "fix: resolve test failure"
git push
```

#### CI is stuck or taking too long

**Cause**: Tests may be running model training or downloading large checkpoints

**Note**: First-time CI runs may take longer due to:
- Downloading model checkpoints from Hugging Face Hub (~1.7GB)
- Installing PyTorch and other large dependencies

**Solution**: Wait for the run to complete. Subsequent runs will be faster due to caching.

### Required Status Checks

Your PR **cannot be merged** until all CI checks pass:
- ✓ Ruff format check
- ✓ Ruff lint check
- ✓ Pyright type check
- ✓ pytest

### Best Practices

1. **Always run pre-commit locally** before pushing to catch issues early
2. **Run baseline tests** before and after changes to prevent regressions
3. **Check CI logs** if a check fails to understand the specific error
4. **Fix violations incrementally** rather than in one large commit
5. **Ask for help** if you're stuck on a CI failure

---

## 7. AI Agent Usage

AI agents (Claude, GitHub Copilot, Gemini, etc.) can help you fix Ruff and pyright violations quickly.

### General Workflow

1. **Commit your code** and encounter violations
2. **Copy the error messages** from pre-commit output
3. **Ask your AI agent** to fix the violations
4. **Review the suggested changes**
5. **Apply the fixes** and re-commit

### Example Prompts

#### Fixing Ruff Violations

**Scenario**: You get Ruff lint errors when committing.

**Pre-commit output**:
```
src/models/vae.py:42:5: F841 Local variable `temp` is assigned to but never used
src/models/vae.py:50:1: D103 Missing docstring in public function
src/models/vae.py:55:20: E225 Missing whitespace around operator
```

**Prompt for Claude/ChatGPT/Gemini**:
```
I got these Ruff linting errors when committing my code:

src/models/vae.py:42:5: F841 Local variable `temp` is assigned to but never used
src/models/vae.py:50:1: D103 Missing docstring in public function
src/models/vae.py:55:20: E225 Missing whitespace around operator

Here's the code:
[paste your code]

Please fix these violations following our coding standards:
- Use Google-style docstrings
- Remove unused variables
- Fix spacing issues
```

#### Adding Type Annotations

**Scenario**: Pyright complains about missing type annotations.

**Pre-commit output**:
```
src/utils/helpers.py:15:5 - error: Type annotation missing for parameter "data"
src/utils/helpers.py:20:5 - error: Return type annotation missing
```

**Prompt for AI agent**:
```
Add type annotations to fix these pyright errors:

src/utils/helpers.py:15:5 - error: Type annotation missing for parameter "data"
src/utils/helpers.py:20:5 - error: Return type annotation missing

Here's the function:
[paste your function]

Context: This function processes PyTorch tensors and returns a dictionary of metrics.
Please add appropriate type hints.
```

#### Adding Docstrings

**Scenario**: Ruff requires docstrings for public functions.

**Pre-commit output**:
```
src/models/predictor.py:100:1: D103 Missing docstring in public function
```

**Prompt for AI agent**:
```
Add a Google-style docstring to this function:

[paste your function]

The function should have:
- Brief one-line summary
- Args section describing parameters
- Returns section describing return value
- Raises section if applicable
```

#### Fixing Import Order

**Scenario**: isort (Ruff rule I) complains about import order.

**Pre-commit output**:
```
src/train_ldm.py:5:1: I001 Import block is un-sorted or un-formatted
```

**Prompt for AI agent**:
```
Fix the import order in this file according to isort rules:

[paste your imports]

Should follow order: standard library, third-party, local imports
```

#### Complex Refactoring

**Scenario**: Function is too complex (McCabe complexity > 10).

**Pre-commit output**:
```
src/data/augmentation.py:50:1: C901 `augment_structure` is too complex (11)
```

**Prompt for AI agent**:
```
This function exceeds maximum complexity (11 > 10):

[paste your function]

Please refactor it by:
1. Breaking it into smaller helper functions
2. Reducing nesting
3. Simplifying conditional logic
4. Maintaining the same functionality
```

### AI Agent Best Practices

1. **Always review AI-generated fixes** - Don't blindly accept changes
2. **Test after applying fixes** - Run `pytest tests/baseline/` to ensure no regressions
3. **Provide context** - Tell the AI agent what the code does
4. **Iterate if needed** - If the first suggestion doesn't work, provide feedback
5. **Learn from fixes** - Understand why the violation occurred to avoid it in future

### GitHub Copilot Integration

If using GitHub Copilot in your editor:

1. **Inline suggestions** - Copilot will often suggest compliant code as you type
2. **Fix violations** - Highlight code with violations and ask Copilot to fix
3. **Generate docstrings** - Type `"""` after a function and let Copilot generate the docstring

### Claude Code CLI (Advanced)

For automated fixing, you can use Claude Code CLI:

```bash
# Install Claude Code CLI
# See: https://claude.com/claude-code

# Run the auto-fix helper script (if available)
./scripts/claude-fix.sh
```

This will:
- Automatically fix Ruff violations
- Run baseline tests to verify no regressions
- Show you a diff of changes for review

---

## Quick Reference

### Essential Commands

```bash
# Setup
./setup-dev.sh                           # Initial setup

# Testing
pytest tests/baseline/ -v                # Run baseline tests
pytest tests/ -v                         # Run all tests
pytest tests/ -m unit                    # Run unit tests only

# Pre-commit
.venv/bin/pre-commit run --all-files    # Run all hooks
.venv/bin/pre-commit autoupdate         # Update hook versions

# Ruff
.venv/bin/ruff format .                 # Format all files
.venv/bin/ruff format --check .         # Check formatting
.venv/bin/ruff check .                  # Lint all files
.venv/bin/ruff check --fix .            # Auto-fix violations

# Pyright
.venv/bin/pyright                       # Run type checking
```

### File Structure

```
chemeleon2/
├── .github/workflows/ci.yml            # CI/CD configuration
├── .pre-commit-config.yaml             # Pre-commit hooks
├── pyproject.toml                      # Project config (Ruff, pyright, pytest)
├── setup-dev.sh                        # Development setup script
├── CONTRIBUTING.md                     # This file
├── src/                                # Source code
└── tests/                              # Tests
    ├── baseline/                       # Baseline validation tests
    ├── unit/                           # Unit tests
    └── integration/                    # Integration tests
```

### Getting Help

1. **Check this guide** - Most common issues are covered in [Troubleshooting](#5-troubleshooting)
2. **Check CI logs** - GitHub Actions logs show detailed error messages
3. **Ask your AI agent** - See [AI Agent Usage](#7-ai-agent-usage) for prompts
4. **Ask the team** - Open an issue or ask in your team chat

---

## Additional Resources

- **Ruff Documentation**: https://docs.astral.sh/ruff/
- **Ruff Rules Reference**: https://docs.astral.sh/ruff/rules/
- **Pre-commit Documentation**: https://pre-commit.com/
- **Pyright Documentation**: https://github.com/microsoft/pyright
- **Google Style Docstrings**: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
- **GitHub Actions**: https://docs.github.com/en/actions
- **pytest Documentation**: https://docs.pytest.org/

---

**Happy coding!** If you have questions or suggestions for improving this guide, please open an issue or submit a PR.
