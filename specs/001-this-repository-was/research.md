# Research: Development Workflow Standards

**Feature**: 001-this-repository-was
**Date**: 2025-10-03
**Status**: Complete

## Overview

This document consolidates research findings and technical decisions for implementing development workflow standards to transition from solo to team development. All decisions were made through the `/clarify` workflow with 12 questions answered.

---

## Decision 1: Code Formatter

**Decision**: Ruff Format
**Rationale**:
- Black-compatible formatting ensures industry-standard style
- 88-character line length is Black's opinionated default
- Ruff combines formatting + linting in single tool (simpler than Black + flake8/pylint)
- Significantly faster than Black (10-100x in benchmarks)
- Python 3.11+ target matches project requirements in pyproject.toml

**Alternatives Considered**:
- **Black**: Standard but separate tool, slower performance
- **autopep8**: PEP 8 compliant but requires configuration, less opinionated
- **YAPF**: Google-style, highly configurable but more complex

**References**:
- Ruff documentation: https://docs.astral.sh/ruff/formatter/
- Black compatibility: https://docs.astral.sh/ruff/formatter/#black-compatibility

---

## Decision 2: Linter Configuration

**Decision**: Ruff with strict rule sets
**Rationale**:
- Comprehensive checks including docstrings, type hints, complexity, security
- Fast execution (critical for < 15 second pre-commit requirement)
- Integrated with formatter (single tool, single configuration)
- Supports auto-fixing for many rules (compatible with AI agent workflow)

**Rule Sets Enabled**:
- **F**: Pyflakes (errors and potential bugs)
- **E/W**: pycodestyle errors and warnings
- **I**: isort (import sorting)
- **N**: pep8-naming conventions
- **D**: pydocstyle (docstring requirements)
- **UP**: pyupgrade (Python version-specific improvements)
- **ANN**: flake8-annotations (type hint coverage)
- **S**: flake8-bandit (security patterns)
- **B**: flake8-bugbear (design problems)
- **C90**: mccabe (complexity limits)

**Alternatives Considered**:
- **Pylint**: Comprehensive but slow, being migrated away from (user removed .pylintrc)
- **flake8**: Popular but requires plugins, slower than Ruff
- **mypy**: Type checking only, doesn't cover all linting needs

**References**:
- Ruff rules documentation: https://docs.astral.sh/ruff/rules/
- Pyproject.toml configuration: https://docs.astral.sh/ruff/configuration/

---

## Decision 3: Pre-commit Framework

**Decision**: pre-commit framework (https://pre-commit.com)
**Rationale**:
- Industry standard for git hooks management
- Language-agnostic, extensible
- Built-in Ruff support via pre-commit hooks
- Supports multiple hooks with defined execution order
- Easy installation and updates

**Hook Configuration**:
1. Ruff format check (fails if formatting needed)
2. Ruff lint check (fails on lint violations)
3. YAML validation for .yaml files
4. TOML validation for .toml files
5. JSON validation for .json files

**Excluded from Pre-commit**:
- pytest (too slow, runs in CI only)
- Complex integration tests

**Alternatives Considered**:
- **Custom git hooks**: Harder to maintain, no version management
- **Husky** (Node.js): Not Python-native
- **Manual hooks**: No standardization across team

**References**:
- Pre-commit documentation: https://pre-commit.com/
- Ruff pre-commit integration: https://github.com/astral-sh/ruff-pre-commit

---

## Decision 4: CI/CD Platform

**Decision**: GitHub Actions
**Rationale**:
- Already using GitHub for repository hosting
- Free for public repositories
- Native integration with Pull Requests
- Required status checks prevent merges on failures
- Matrix builds for multi-platform testing if needed

**Workflow Steps**:
1. Check out code
2. Set up Python 3.11+
3. Install dependencies (including Ruff, pytest)
4. Run Ruff format check
5. Run Ruff lint check
6. Run pytest (full test suite)
7. Report results to PR

**Alternatives Considered**:
- **GitLab CI**: Not applicable (using GitHub)
- **Travis CI**: Legacy, less active development
- **CircleCI**: Additional service, unnecessary complexity

**References**:
- GitHub Actions documentation: https://docs.github.com/en/actions
- Python workflow examples: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python

---

## Decision 5: Setup Automation

**Decision**: Single-command setup script
**Rationale**:
- Git hooks cannot auto-install on clone (security restriction)
- One-command experience is developer-friendly
- Script can be simple: `make setup` or `./setup.sh` or `python setup.py dev`
- Installs pre-commit hooks + dependencies in single step

**Setup Script Responsibilities**:
1. Check Python version (>= 3.11)
2. Install pre-commit framework (`pip install pre-commit`)
3. Install Ruff (`pip install ruff` or via requirements-dev.txt)
4. Install pre-commit hooks (`pre-commit install`)
5. Optionally run pre-commit on all files for validation

**Alternatives Considered**:
- **Manual README instructions**: Error-prone, inconsistent setup
- **Git template hooks**: Complex, not portable
- **Docker-based setup**: Overkill for development tooling

**References**:
- Pre-commit installation: https://pre-commit.com/#install

---

## Decision 6: File Type Coverage

**Decision**: .py, .yaml, .toml, .json
**Rationale**:
- **Python (.py)**: Primary source code
- **YAML (.yaml)**: Configuration files (e.g., .pre-commit-config.yaml, CI workflows)
- **TOML (.toml)**: Configuration files (pyproject.toml, Ruff config)
- **JSON (.json)**: Data and configuration files

**Validation Approach**:
- Python: Ruff format + lint
- YAML: Pre-commit check-yaml hook
- TOML: Pre-commit check-toml hook
- JSON: Pre-commit check-json hook

**Excluded**:
- Markdown (.md): Documentation, no strict formatting needed
- Jupyter Notebooks (.ipynb): Special case, may add later if needed

---

## Decision 7: Enforcement Policy

**Decision**: Block commits until fixed (manually or via AI agent)
**Rationale**:
- Prevents non-compliant code from entering repository
- Maintains zero technical debt policy
- AI agent support provides escape hatch for complex fixes
- Developers can fix locally before committing

**Workflow**:
1. Developer commits → Pre-commit runs
2. If violations → Commit blocked with error messages
3. Developer options:
   - Fix manually and re-commit
   - Ask AI agent (Claude, Codex, Gemini) to fix
   - Skip with `--no-verify` (discouraged, CI will catch)

**Alternatives Considered**:
- **Warn only**: Would accumulate technical debt
- **Auto-fix and commit**: Loss of developer control, potential unwanted changes

---

## Decision 8: Retroactive Application Strategy

**Decision**: Apply to all existing code with coordination
**Rationale**:
- Establishes baseline for all future development
- Prevents "grandfathered" non-compliant code
- One-time effort vs ongoing mixed standards

**Coordination Strategy**:
1. Announce retroactive formatting to team
2. Merge or shelve active PRs before formatting
3. Create dedicated PR for formatting changes only
4. Use `ruff format .` and `ruff check --fix .` on entire codebase
5. Commit as single "formatting: apply Ruff standards to codebase"
6. Merge to main
7. Team rebases feature branches after merge

**Risk Mitigation**:
- Large diff but merge conflicts minimal (formatting-only changes)
- Clear commit message and PR description
- Can rollback if issues discovered

**Alternatives Considered**:
- **Gradual migration**: "Touch files only when modified" - creates inconsistency
- **New code only**: Leaves existing code non-compliant

---

## Decision 9: Documentation Scope

**Decision**: Comprehensive CONTRIBUTING.md with examples
**Rationale**:
- Team onboarding requires clear instructions
- AI agent usage examples help developers leverage automation
- Troubleshooting guide reduces friction

**CONTRIBUTING.md Sections**:
1. **Setup Instructions**: One-command setup, prerequisites
2. **Development Workflow**: Commit process, pre-commit expectations
3. **Coding Standards**: Ruff rules, line length, type hints policy
4. **AI Agent Usage**: How to ask Claude/Codex/Gemini to fix violations
5. **Troubleshooting**: Common issues (hook failures, version mismatches)
6. **CI/CD**: How GitHub Actions works, required checks

**Inline Documentation**:
- pyproject.toml: Comments explaining Ruff rule choices
- .pre-commit-config.yaml: Comments on hook order

---

## Decision 10: Performance Constraints

**Decision**: < 15 second pre-commit validation
**Rationale**:
- Balance between thoroughness and developer productivity
- Ruff is fast enough for large codebases
- Excluding pytest keeps validation quick

**Optimization Strategies**:
- Ruff runs only on staged files (not entire codebase)
- Parallel hook execution where possible
- File-specific hooks (YAML check only on .yaml files)

**Monitoring**:
- Measure actual pre-commit times during implementation
- Adjust hook configuration if exceeds 15 seconds

---

## Decision 11: AI Agent Integration

**Decision**: Support multiple AI agents (Claude, Codex, Gemini)
**Rationale**:
- Developers may have different AI tool preferences
- Ruff output is structured and parseable by all AI agents
- Auto-fixing capability reduces manual burden

**Integration Approach**:
- No custom tooling needed
- Developers copy/paste Ruff errors to AI agent
- AI agent suggests fixes, developer reviews and applies
- Document common prompts in CONTRIBUTING.md

**Example Prompts** (for documentation):
- "Fix these Ruff violations: [paste errors]"
- "Add docstrings to pass Ruff checks"
- "Add type hints to these functions: [paste errors]"

---

## Technical Stack Summary

| Component | Technology | Version/Config |
|-----------|-----------|----------------|
| Formatter | Ruff Format | Latest stable |
| Linter | Ruff | Strict rule sets (F,E,W,I,N,D,UP,ANN,S,B,C90) |
| Pre-commit | pre-commit framework | 3.x |
| CI/CD | GitHub Actions | ubuntu-latest, Python 3.11+ |
| Testing | pytest | CI only (not pre-commit) |
| Documentation | Markdown (CONTRIBUTING.md) | Comprehensive with examples |
| Setup | Shell/Python script | One-command installation |

---

## Implementation Considerations

### Ruff Configuration Location
- Use `pyproject.toml` (already exists in repo)
- Extend existing `[project]` section with `[tool.ruff]`

### Pre-commit Configuration
- `.pre-commit-config.yaml` at repository root
- Specify Ruff version to ensure consistency between local and CI

### CI Configuration
- `.github/workflows/ci.yml` (new file)
- Run on all PRs and pushes to main
- Use `ruff-action` for GitHub Actions integration

### Version Pinning
- Pin Ruff version in both pre-commit config and CI
- Update periodically but consciously (avoid breaking changes mid-sprint)

---

## Decision 12: Baseline Testing Strategy (CRITICAL)

**Decision**: Create baseline tests BEFORE applying formatting changes
**Rationale**:
- **User Insight**: "Before applying lint/format updates, shouldn't we test that the code works properly first?"
- This follows Test-Driven Development principles
- Prevents silent regressions during retroactive formatting
- No existing tests in `tests/` directory - must create baseline first

**Baseline Test Types**:

1. **Smoke Tests**: Quick sanity checks without full training
   - Model instantiation tests (VAE, LDM, RL modules)
   - Forward pass shape validation
   - DataLoader batching verification

2. **Critical Validation Test**: Overfit Single Batch
   - As per Andrej Karpathy: "If you can't overfit on a tiny batch, things are definitely broken"
   - Train model on single batch for 100-2000 iterations
   - Assert final_loss < initial_loss * 0.1
   - This catches gradient bugs, incorrect loss functions, broken backprop

3. **Data Pipeline Tests**:
   - Verify dataloader produces correct tensor shapes (BxCxHxW)
   - Verify labels match image count
   - Verify dtypes (float32 for images, long for labels)

**pytest Configuration**:
```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = ["-ra", "-q", "--strict-markers"]
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
    "smoke: quick sanity checks for core functionality",
    "baseline: baseline validation tests",
    "unit: isolated component tests",
    "integration: integration tests",
    "slow: tests that take >1s",
]
```

**Test Directory Structure**:
```
tests/
├── conftest.py              # Shared fixtures (device, dummy datasets)
├── baseline/                # Baseline validation (run before/after formatting)
│   ├── test_vae_module.py
│   ├── test_ldm_module.py
│   ├── test_rl_module.py
│   └── test_data_loading.py
├── integration/             # Future integration tests
└── unit/                    # Future unit tests
```

**Implementation Workflow**:
1. ✅ Research complete (including pytest best practices)
2. → Create `tests/` structure with baseline tests
3. → Run tests and verify they pass (green baseline)
4. → Apply Ruff formatting to codebase (`ruff format .`)
5. → Re-run tests and verify still green (no regressions)
6. → Commit formatted code with confidence

**Key Pytest Patterns for ML**:
- Use `fast_dev_run=True` for PyTorch Lightning smoke tests
- Use `torch.testing.assert_close()` for tensor comparisons (not `==`)
- Set random seeds in fixtures for reproducibility
- Exclude pytest from pre-commit hooks (too slow, run in CI only)
- Use pytest markers to organize test types (`pytest -m smoke`)

**Alternatives Considered**:
- **No baseline tests**: Too risky - formatting might break code silently
- **Manual validation**: Not reproducible or scalable for team
- **Full test suite first**: Too time-consuming for initial validation

**References**:
- pytest documentation: https://docs.pytest.org/
- PyTorch Lightning fast_dev_run: https://lightning.ai/docs/pytorch/stable/common/trainer.html#fast-dev-run
- ML testing best practices: https://madewithml.com/courses/mlops/testing/

---

## Decision 13: Pyright Type Checking Mode

**Decision**: Use pyright in basic mode
**Rationale**:
- **User Feedback**: "strict mode is not realistic for current codebase"
- Basic mode provides essential type checking without overwhelming strictness
- Allows gradual type hint adoption without blocking development
- Catches common type errors while being practical for ML/scientific code
- Reduces configuration complexity by removing granular warning settings

**Configuration**:
```toml
[tool.pyright]
typeCheckingMode = "basic"
pythonVersion = "3.11"
```

**Alternatives Considered**:
- **Strict mode**: Too demanding for existing codebase, would require extensive refactoring
- **Off mode**: No type checking, loses valuable static analysis benefits
- **Standard mode**: Basic provides better balance for this project

**References**:
- Pyright documentation: https://github.com/microsoft/pyright/blob/main/docs/configuration.md

---

## Open Questions (if any)

**None** - All questions resolved through `/clarify` workflow + user insight on baseline testing.

---

## References

- Ruff Documentation: https://docs.astral.sh/ruff/
- Pre-commit Documentation: https://pre-commit.com/
- GitHub Actions: https://docs.github.com/en/actions
- Python Packaging Guide (pyproject.toml): https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

---

**Research Complete** ✅
All technical decisions made and documented. Ready for Phase 1: Design & Contracts.
