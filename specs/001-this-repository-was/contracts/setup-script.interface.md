# Contract: Setup Script Interface

**Location**: Repository root (e.g., `setup-dev.sh` or `Makefile` with `setup` target)

**Purpose**: One-command developer environment setup

---

## Command Interface

### Option 1: Shell Script
```bash
./setup-dev.sh [OPTIONS]
```

### Option 2: Makefile Target
```bash
make setup
```

### Option 3: Python Script
```bash
python setup.py dev
```

**Choose one implementation approach - all must satisfy same contract**

---

## Required Behavior

### 1. Prerequisites Check
```
MUST check Python version >= 3.11
→ If failed: Print error message, exit code 1
```

### 2. Install Pre-commit Framework
```
MUST run: pip install pre-commit
→ If already installed: Skip or upgrade
→ If failed: Print error message, exit code 2
```

### 3. Install Ruff
```
MUST run: pip install ruff
→ If already installed: Skip or upgrade
→ If failed: Print error message, exit code 2
```

### 4. Install Pre-commit Hooks
```
MUST run: pre-commit install
→ Creates .git/hooks/pre-commit
→ If failed: Print error message, exit code 3
```

### 5. Optional Validation
```
OPTIONAL: Run pre-commit on all files for initial validation
COMMAND: pre-commit run --all-files
→ If failed: Warn but exit 0 (setup succeeded, code needs fixing)
```

---

## Exit Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 0 | Success | Setup completed successfully |
| 1 | Python version error | Python < 3.11 detected |
| 2 | Installation error | pip install failed |
| 3 | Hook installation error | pre-commit install failed |

---

## Output Requirements

### Successful Output Example
```
Checking Python version... ✓ Python 3.11.5
Installing pre-commit... ✓ Done
Installing Ruff... ✓ Done
Installing pre-commit hooks... ✓ Done

✓ Development environment setup complete!

Next steps:
  1. Make your changes
  2. Commit (pre-commit will run automatically)
  3. Push and open a PR
```

### Error Output Example
```
Checking Python version... ✗ Found Python 3.9.7
ERROR: Python 3.11+ required. Please upgrade Python.

Installation aborted.
```

---

## Idempotency

**MUST be safe to run multiple times**:
- Re-running should update/reinstall tools
- No errors if already installed
- No side effects on repeated runs

---

## Contract Test: test_setup_script.py

Validates:
1. Script exists and is executable
2. Running with Python 3.11+ returns exit code 0
3. Running with Python < 3.11 returns exit code 1 (simulate with version check mock)
4. After successful run, `.git/hooks/pre-commit` exists
5. After successful run, `pre-commit --version` works
6. After successful run, `ruff --version` works
7. Idempotency: Running twice succeeds both times
8. Output includes success messages

---

## Implementation Notes

- Use `set -e` in shell scripts (exit on first error)
- Check commands exist before running (`command -v pre-commit`)
- Provide clear error messages with actionable fixes
- Document script usage in CONTRIBUTING.md

---

**Contract Complete** ✅
Ready for test-driven implementation.
