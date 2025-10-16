# Contributing to chemeleon2

Welcome to the chemeleon2 project! This guide will help you set up your development environment and understand our workflow.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Setup Instructions](#2-setup-instructions)
3. [Git Workflow](#3-git-workflow)
4. [Development Workflow](#4-development-workflow)
5. [Testing](#5-testing)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

Before starting, ensure you have:

- **Python 3.11+** installed
- **Git** installed
- **uv** package manager (recommended)

### Installing uv

If you don't have `uv` installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 2. Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/hspark1212/chemeleon2.git
cd chemeleon2
```

### Step 2: Install Dependencies

```bash
# Install all dependencies with  dev dependencies (pytest, ruff, pyright, etc.)
uv sync --extra dev

# Optional install torch with your cuda version (e.g., for CUDA 12.8)
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install Pre-commit Hooks

Install pre-commit hooks to ensure code quality on every commit:

```bash
# activate the virtual environment
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

### Step 4: Verify Installation

Test that everything works:

```bash
# Run pre-commit hooks on all files to ensure code quality
pre-commit run --all-files

# Run pytest to ensure tests pass
pytest -v
```

---

## 3. Git Workflow

### Key Concepts

- **Repository (repo)**: Folder with your project code and its entire history
- **Branch**: Independent workspace for changes. `main` contains stable code
- **Commit**: Saved snapshot of your code changes
- **Push**: Upload local changes to GitHub
- **Pull Request (PR)**: Request to merge your changes into `main`

### Workflow Steps

**⚠️ NEVER work directly on `main` branch!**

1. **Create a new branch** for your work
2. **Make your changes** and commit them
3. **Push your branch** to GitHub
4. **Create a Pull Request**
5. **Wait for review** - maintainers will review and merge

### Common Git Commands

```bash
# Check what branch you're on
git branch

# Create and switch to a new branch
git checkout -b feature/my-feature

# Check what files changed
git status

# Stage all changes
git add .

# Commit (pre-commit hooks run automatically)
git commit -m "feat: add new feature"

# Push to GitHub (first time)
git push -u origin feature/my-feature

# Push to GitHub (subsequent times)
git push
```

### Branch Naming Convention

- Features: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`
- Examples: `feature/add-model`, `fix/training-bug`, `docs/update-readme`

---

## 4. Development Workflow

### Making Changes

1. **Activate the virtual environment** (if not already activated):
   ```bash
   source .venv/bin/activate
   ```

2. **Create a branch** (see Git Workflow above)

3. **Make your changes** in your code editor

4. **Test your changes locally:**
   ```bash
   # Run relevant tests
   pytest tests/

   # Or run specific test files
   pytest tests/unit/test_your_feature.py
   ```

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: descriptive commit message"
   ```

   Pre-commit hooks will automatically run. If they fail:
   - Fix the issues shown
   - Stage changes again: `git add .`
   - Commit again: `git commit -m "your message"`

6. **Push to GitHub:**
   ```bash
   git push -u origin your-branch-name
   ```

7. **Create a Pull Request:**
   - Go to GitHub
   - Click "Compare & pull request"
   - Describe your changes
   - Submit for review

### Commit Message Format

Use conventional commit format:

- `feat: add new feature` - New functionality
- `fix: resolve bug in training` - Bug fixes
- `docs: update README` - Documentation
- `test: add unit tests` - Tests
- `refactor: reorganize code` - Code restructuring
- `chore: update dependencies` - Maintenance

---

## 5. Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m baseline       # Baseline validation
pytest tests/ -m unit          # Unit tests
pytest tests/ -m integration   # Integration tests

# Run specific test file
pytest tests/unit/test_featurizer.py -v
```

### Running Linters

```bash
# Check code style
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .
```

### Pre-commit Hooks

Hooks run automatically on `git commit`. To run manually:

```bash
# Run on all files
.venv/bin/pre-commit run --all-files

# Run on staged files only
.venv/bin/pre-commit run
```

---

## 6. Troubleshooting

### Pre-commit Hooks Not Running

```bash
# Reinstall hooks
.venv/bin/pre-commit uninstall
.venv/bin/pre-commit install
```

### Import Errors

```bash
# Reinstall in editable mode
uv pip install -e ".[dev]"
```

### Dependency Issues

```bash
# Sync dependencies
uv sync

# Clear cache and reinstall
rm -rf .venv
uv sync
uv pip install -e ".[dev]"
.venv/bin/pre-commit install
```

### Python Version Issues

Ensure you're using Python 3.11+:
```bash
python --version
# or
.venv/bin/python --version
```

---

Thank you for contributing to chemeleon2! 🎉
