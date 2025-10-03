# Tasks: Development Workflow Standards for Team Collaboration

**Input**: Design documents from `/home/hyunsoo/VScodeProjects/chemeleon2/specs/001-this-repository-was/`
**Prerequisites**: [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/)

## Execution Flow (main)
```
1. Load plan.md from feature directory ✅
   → Extract: Python 3.11+, Ruff, pre-commit, pytest, GitHub Actions
2. Load optional design documents ✅
   → data-model.md: Configuration entities identified
   → contracts/: 4 contract files found
   → research.md: All technical decisions resolved
3. Generate tasks by category ✅
   → Setup: Project init, test infrastructure
   → Tests: Baseline tests FIRST (critical requirement)
   → Core: Configuration files, scripts
   → Integration: Pre-commit hooks, CI/CD
   → Polish: Documentation, verification
4. Apply task rules ✅
   → Test files = mark [P] for parallel
   → Configuration files = mark [P] for parallel
   → Same file modifications = sequential
5. Number tasks sequentially (T001-T027) ✅
6. Generate dependency graph ✅
7. Create parallel execution examples ✅
8. Validate task completeness ✅
   → All contracts have tests ✅
   → All entities have configuration tasks ✅
   → Baseline tests BEFORE formatting changes ✅
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- Single Python project structure
- `src/` at repository root (existing ML code)
- `tests/` at repository root (new test infrastructure)
- Configuration files at repository root

---

## Phase 3.1: Setup & Infrastructure

- [x] **T001** Create test directory structure
  - Create `tests/` directory
  - Create `tests/baseline/` subdirectory
  - Create `tests/contract/` subdirectory
  - Create `tests/integration/` subdirectory (placeholder for future)
  - Create `tests/unit/` subdirectory (placeholder for future)
  - Create `tests/conftest.py` for shared fixtures

- [ ] **T002** Configure pytest in pyproject.toml
  - Add `[tool.pytest.ini_options]` section to `pyproject.toml`
  - Set `minversion = "7.0"`
  - Set `testpaths = ["tests"]`
  - Set `pythonpath = ["src"]`
  - Add markers: smoke, baseline, unit, integration, slow
  - Add `addopts = ["-ra", "-q", "--strict-markers"]`

---

## Phase 3.2: Baseline Tests (CRITICAL - Must Pass BEFORE Formatting)

⚠️ **CRITICAL**: These tests establish a functional baseline to prevent silent regressions during formatting changes. All tests must be written and PASS before proceeding to Phase 3.3.

- [ ] **T003 [P]** Create baseline test for VAE module in `tests/baseline/test_vae_module.py`
  - Test VAE model instantiation
  - Test forward pass shape validation
  - Test overfit-single-batch capability
  - Mark with `@pytest.mark.smoke` and `@pytest.mark.baseline`

- [ ] **T004 [P]** Create baseline test for LDM module in `tests/baseline/test_ldm_module.py`
  - Test LDM model instantiation
  - Test forward pass shape validation
  - Test overfit-single-batch capability
  - Mark with `@pytest.mark.smoke` and `@pytest.mark.baseline`

- [ ] **T005 [P]** Create baseline test for RL module in `tests/baseline/test_rl_module.py`
  - Test RL agent instantiation
  - Test policy forward pass
  - Test overfit-single-batch capability
  - Mark with `@pytest.mark.smoke` and `@pytest.mark.baseline`

- [ ] **T006 [P]** Create baseline test for data loading in `tests/baseline/test_data_loading.py`
  - Test dataloader batching (verify shapes BxCxHxW)
  - Test label alignment (count matches image count)
  - Test dtypes (float32 for images, long for labels)
  - Mark with `@pytest.mark.smoke` and `@pytest.mark.baseline`

- [ ] **T007** Run baseline tests and verify GREEN status
  - Execute: `pytest tests/baseline/ -v -m baseline`
  - All 4 test files must pass
  - If failures: Fix code issues BEFORE proceeding
  - Document baseline test results (copy output to feature docs)
  - **GATE**: Must pass before T008-T014

---

## Phase 3.3: Configuration Files (Only After Baseline Tests Pass)

- [ ] **T008 [P]** Configure Ruff in pyproject.toml
  - Add `[tool.ruff]` section to existing `pyproject.toml`
  - Set `target-version = "py311"`
  - Set `line-length = 88`
  - Add `[tool.ruff.lint]` section
  - Set `select = ["F", "E", "W", "I", "N", "D", "UP", "ANN", "S", "B", "C90"]`
  - Add `[tool.ruff.lint.pydocstyle]` with `convention = "google"`
  - Add `[tool.ruff.lint.mccabe]` with `max-complexity = 10`
  - Add inline comments explaining each rule category (e.g., "# F: Pyflakes errors", "# E/W: PEP 8 style", "# D: Docstrings", etc.) per FR-008

- [ ] **T009 [P]** Create pre-commit configuration in `.pre-commit-config.yaml`
  - Add ruff-pre-commit repo with specific pinned version (e.g., `v0.1.9`, NOT `latest` or branch refs) to ensure local/CI consistency per edge case spec.md:L89
  - Configure ruff-format hook
  - Configure ruff lint hook with `--fix` arg
  - Add pre-commit-hooks repo with YAML/TOML/JSON validators (also pinned)
  - Ensure ruff-format runs before ruff lint
  - Exclude pytest from pre-commit hooks

- [ ] **T010 [P]** Create GitHub Actions workflow in `.github/workflows/ci.yml`
  - Set name: "CI"
  - Trigger on pull_request to main/develop branches
  - Trigger on push to main branch
  - Configure ubuntu-latest runner
  - Add steps: checkout, setup Python 3.11, install deps
  - Add step: `ruff format --check .`
  - Add step: `ruff check .`
  - Add step: `pytest` (runs full test suite including baseline)

- [ ] **T011 [P]** Create setup script in `setup-dev.sh`
  - Add shebang and `set -e`
  - Check Python version >= 3.11 (exit 1 if failed)
  - Install pre-commit framework (exit 2 if failed)
  - Install Ruff (exit 2 if failed)
  - Run `pre-commit install` (exit 3 if failed)
  - Optionally run `pre-commit run --all-files` (warn but exit 0)
  - Make script executable: `chmod +x setup-dev.sh`

---

## Phase 3.4: Contract Tests (Validate Configuration Contracts)

- [ ] **T012 [P]** Contract test for Ruff config in `tests/contract/test_ruff_config.py`
  - Assert `pyproject.toml` exists
  - Assert valid TOML syntax
  - Assert `[tool.ruff]` section exists
  - Assert `target-version = "py311"`
  - Assert `line-length = 88`
  - Assert all 11 rule sets in select array

- [ ] **T013 [P]** Contract test for pre-commit config in `tests/contract/test_precommit_config.py`
  - Assert `.pre-commit-config.yaml` exists
  - Assert valid YAML syntax
  - Assert ruff-pre-commit repo present
  - Assert ruff-format and ruff hooks present
  - Assert YAML/TOML/JSON validators present
  - Assert versions are pinned (not "latest")
  - Assert pytest NOT in hooks

- [ ] **T014 [P]** Contract test for GitHub Actions in `tests/contract/test_github_actions.py`
  - Assert `.github/workflows/ci.yml` exists
  - Assert valid YAML syntax
  - Assert triggers on pull_request
  - Assert Python 3.11+ in setup step
  - Assert ruff format check step present
  - Assert ruff lint check step present
  - Assert pytest step present

- [ ] **T015 [P]** Contract test for setup script in `tests/contract/test_setup_script.py`
  - Assert `setup-dev.sh` exists and is executable
  - Test exit code 0 with Python 3.11+
  - Test creates `.git/hooks/pre-commit` after successful run
  - Test idempotency (running twice succeeds)
  - Verify output includes success messages

---

## Phase 3.5: Application & Verification

- [ ] **T016** Apply Ruff formatting to existing codebase
  - Run: `ruff format .` from repository root
  - Review changes (should only affect code style)
  - Commit as single "formatting: apply Ruff standards to codebase"
  - **Dependency**: Requires T008 (Ruff configuration)

- [ ] **T017** Apply Ruff auto-fixes to existing codebase
  - Run: `ruff check --fix .` from repository root
  - Review auto-fixed violations
  - Commit as "lint: apply Ruff auto-fixes"
  - **Dependency**: Requires T016 (formatting first)

- [ ] **T018** Re-run baseline tests after formatting changes
  - Execute: `pytest tests/baseline/ -v -m baseline`
  - All tests must still PASS (verify no regressions)
  - Compare results with T007 output
  - If failures: Formatting broke code - rollback and investigate
  - **GATE**: Must pass before proceeding
  - **Dependency**: Requires T016, T017

- [ ] **T019** Manual fixes for remaining Ruff violations
  - Run: `ruff check .` to identify remaining issues
  - Fix violations that couldn't be auto-fixed (e.g., missing docstrings, type hints)
  - Re-run `ruff check .` until clean
  - Commit as "lint: manually fix remaining Ruff violations"
  - **Dependency**: Requires T017, T018

---

## Phase 3.6: Integration & Validation

- [ ] **T020** Install and test pre-commit hooks locally
  - Run: `./setup-dev.sh`
  - Verify exit code 0
  - Verify `.git/hooks/pre-commit` created
  - Test hook with intentional violation (should block commit)
  - Fix violation and verify commit succeeds
  - **Dependency**: Requires T009, T011, T019 (clean codebase)

- [ ] **T021** Run all contract tests
  - Execute: `pytest tests/contract/ -v`
  - All 4 contract tests must pass
  - **Dependency**: Requires T008-T011 (all configs created)

- [ ] **T022** Verify CI/CD workflow in GitHub
  - Push branch to GitHub
  - Open test Pull Request
  - Verify GitHub Actions CI runs automatically
  - Verify all checks pass (format, lint, pytest)
  - Close test PR
  - **Dependency**: Requires T010, T019 (clean codebase), T021 (tests pass)

---

## Phase 3.7: Claude Agent Auto-Fix Integration

- [ ] **T023 [P]** Create Claude agent pre-commit hook script in `.git-hooks/claude-autofix.sh`
  - Create custom git hooks directory `.git-hooks/`
  - Write bash script that:
    - Runs `ruff format --check .` and `ruff check .`
    - Captures Ruff error output
    - If errors found, invokes Claude Code CLI with error messages
    - Applies Claude's fixes automatically
    - Re-runs Ruff validation
    - Returns exit code 0 if fixed, 1 if still failing
  - Make script executable: `chmod +x .git-hooks/claude-autofix.sh`
  - **Dependency**: Requires T019 (clean baseline)

- [ ] **T024 [P]** Create Claude agent configuration in `.claude/hooks/pre-commit-autofix.md`
  - Create `.claude/hooks/` directory
  - Define hook prompt template for Ruff error fixing
  - Include instructions for Claude to:
    - Parse Ruff error messages
    - Apply formatting fixes
    - Add missing docstrings
    - Add missing type hints
    - Fix import ordering
    - Reduce complexity if needed
  - Specify output format (direct file edits)

- [ ] **T025** Update pre-commit config to support Claude auto-fix
  - Modify `.pre-commit-config.yaml` to add optional manual hook
  - Add comment explaining Claude auto-fix activation
  - Document opt-in mechanism (developers can enable/disable)
  - **Dependency**: Requires T009, T023, T024

- [ ] **T026** Create Claude auto-fix wrapper command in `scripts/claude-fix.sh`
  - Create `scripts/` directory if not exists
  - Write wrapper script that:
    - Checks if Claude Code CLI is installed
    - Runs pre-commit checks
    - If failures, invokes `.git-hooks/claude-autofix.sh`
    - Shows before/after diff
    - Asks user to confirm changes
  - Make executable: `chmod +x scripts/claude-fix.sh`
  - **Dependency**: Requires T023

---

## Phase 3.8: Documentation

- [ ] **T027** Create CONTRIBUTING.md at repository root
  - Add "Table of Contents" section
  - Add "Setup Instructions" with one-command setup
  - Add "Development Workflow" with commit process
  - Add "Coding Standards" explaining Ruff rules
  - Add "Manual AI Agent Usage" with example prompts for copying errors
  - Add "Claude Auto-Fix Usage" explaining `./scripts/claude-fix.sh` command
  - Add "Troubleshooting" for common issues
  - Add "CI/CD" explaining GitHub Actions workflow
  - Add "CI/CD Failure Troubleshooting" explaining how to interpret GitHub Actions check failures (format check, lint check, pytest failures) for developers new to the workflow
  - Reference quickstart.md for validation steps
  - **Dependency**: Requires T020 (verified workflow), T026 (Claude integration)

---

## Dependencies Graph

```
Setup Phase:
T001 (test structure) → T002 (pytest config) → T003-T006 (baseline tests)
T003-T006 (baseline tests) → T007 (verify baseline)

T007 GATES → Configuration Phase

Configuration Phase:
T008 (Ruff config) ┐
T009 (pre-commit)  ├─ [Parallel - different files]
T010 (GitHub CI)   │
T011 (setup script)┘

Contract Tests:
T012 (Ruff contract)      ┐
T013 (pre-commit contract)├─ [Parallel - different test files]
T014 (GitHub contract)    │
T015 (setup contract)     ┘

Application Phase:
T008 → T016 (format) → T017 (auto-fix) → T018 (verify baseline) → T019 (manual fixes)

T018 GATES → Integration Phase

Integration Phase:
T019 → T020 (install hooks)
T008-T011 → T021 (contract tests)
T019, T021 → T022 (verify CI)

Claude Auto-Fix Phase:
T019 → T023 (Claude hook script) ┐
T019 → T024 (Claude config)      ├─ [Parallel - different files]
                                 │
T009, T023, T024 → T025 (update pre-commit)
T023 → T026 (wrapper script)

Documentation:
T020, T026 → T027 (CONTRIBUTING.md)
```

---

## Parallel Execution Examples

### Example 1: Baseline Tests (T003-T006)
```bash
# All baseline tests can run in parallel - different files, no dependencies
pytest tests/baseline/ -n auto  # Using pytest-xdist for parallel execution

# Or launch as separate tasks:
# Task 1: "Create baseline test for VAE module in tests/baseline/test_vae_module.py"
# Task 2: "Create baseline test for LDM module in tests/baseline/test_ldm_module.py"
# Task 3: "Create baseline test for RL module in tests/baseline/test_rl_module.py"
# Task 4: "Create baseline test for data loading in tests/baseline/test_data_loading.py"
```

### Example 2: Configuration Files (T008-T011)
```bash
# All configuration files can be created in parallel - different files
# Task 1: "Configure Ruff in pyproject.toml"
# Task 2: "Create pre-commit configuration in .pre-commit-config.yaml"
# Task 3: "Create GitHub Actions workflow in .github/workflows/ci.yml"
# Task 4: "Create setup script in setup-dev.sh"
```

### Example 3: Contract Tests (T012-T015)
```bash
# All contract tests can run in parallel - different test files
pytest tests/contract/ -n auto

# Or launch as separate tasks:
# Task 1: "Contract test for Ruff config in tests/contract/test_ruff_config.py"
# Task 2: "Contract test for pre-commit config in tests/contract/test_precommit_config.py"
# Task 3: "Contract test for GitHub Actions in tests/contract/test_github_actions.py"
# Task 4: "Contract test for setup script in tests/contract/test_setup_script.py"
```

### Example 4: Claude Auto-Fix Components (T023-T024)
```bash
# Claude hook script and config can be created in parallel - different files
# Task 1: "Create Claude agent pre-commit hook script in .git-hooks/claude-autofix.sh"
# Task 2: "Create Claude agent configuration in .claude/hooks/pre-commit-autofix.md"
```

---

## Critical Success Gates

### Gate 1: Baseline Tests (After T007)
- **Requirement**: All baseline tests MUST PASS
- **Action if Failed**: Fix code issues, do NOT proceed to configuration
- **Rationale**: Establishes functional baseline before formatting changes

### Gate 2: Post-Formatting Verification (After T018)
- **Requirement**: Baseline tests MUST STILL PASS after formatting
- **Action if Failed**: Rollback formatting changes, investigate regressions
- **Rationale**: Proves formatting didn't break functionality

---

## Notes

- **[P]** tasks indicate different files with no dependencies - safe for parallel execution
- **Sequential tasks** (T016 → T017 → T018 → T019) must run in order
- **Commit frequently**: After each task or logical group of parallel tasks
- **Test-first approach**: Baseline tests created and verified BEFORE any formatting changes
- **Avoid**: Vague tasks, modifying same file in parallel tasks
- **Performance target**: Pre-commit validation < 15 seconds, CI pipeline < 5 minutes
- **Claude Auto-Fix**: Optional feature - developers can use `./scripts/claude-fix.sh` to automatically fix Ruff violations instead of manual copy-paste to AI

---

## Validation Checklist

*GATE: Checked before marking tasks complete*

- [x] All contracts have corresponding tests (4 contracts → 4 contract tests)
- [x] All configuration entities have creation tasks (Ruff, pre-commit, CI, setup)
- [x] All tests come before implementation (baseline tests → config → application)
- [x] Parallel tasks truly independent (different files, verified)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Critical requirement met: Baseline tests BEFORE formatting changes (T007 gates T008-T014)
- [x] Test-First Development: T003-T007 before T016-T019
- [x] Claude Auto-Fix integration added: T023-T027 for automatic Ruff violation fixing

---

**Tasks Complete** ✅
Total: 27 tasks (23 original + 4 Claude Auto-Fix + 1 updated documentation)
Ready for Phase 3 execution. Begin with T001.
