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

- [x] **T002** Configure pytest in pyproject.toml
  - Add `[tool.pytest.ini_options]` section to `pyproject.toml`
  - Set `minversion = "7.0"`
  - Set `testpaths = ["tests"]`
  - Set `pythonpath = ["src"]`
  - Add markers: smoke, baseline, unit, integration, slow
  - Add `addopts = ["-ra", "-q", "--strict-markers"]`

---

## Phase 3.2: Baseline Tests (CRITICAL - Must Pass BEFORE Formatting)

⚠️ **CRITICAL**: These tests establish a functional baseline to prevent silent regressions during formatting changes. All tests must be written and PASS before proceeding to Phase 3.3.

- [x] **T003 [P]** Create baseline test for VAE module in `tests/baseline/test_vae_module.py`
  - Test VAE model instantiation
  - Test forward pass shape validation
  - Test overfit-single-batch capability
  - Mark with `@pytest.mark.smoke` and `@pytest.mark.baseline`

- [x] **T004 [P]** Create baseline test for LDM module in `tests/baseline/test_ldm_module.py`
  - Test LDM model instantiation
  - Test forward pass shape validation
  - Test overfit-single-batch capability
  - Mark with `@pytest.mark.smoke` and `@pytest.mark.baseline`

- [x] **T005 [P]** Create baseline test for RL module in `tests/baseline/test_rl_module.py`
  - Test RL agent instantiation
  - Test policy forward pass
  - Test overfit-single-batch capability
  - Mark with `@pytest.mark.smoke` and `@pytest.mark.baseline`

- [x] **T006 [P]** Create baseline test for data loading in `tests/baseline/test_data_loading.py`
  - Test dataloader batching (verify shapes BxCxHxW)
  - Test label alignment (count matches image count)
  - Test dtypes (float32 for images, long for labels)
  - Mark with `@pytest.mark.smoke` and `@pytest.mark.baseline`

- [x] **T007** Run baseline tests and verify GREEN status
  - Execute: `pytest tests/baseline/ -v -m baseline`
  - All 4 test files must pass
  - If failures: Fix code issues BEFORE proceeding
  - Document baseline test results (copy output to feature docs)
  - **GATE**: Must pass before T008-T014

---

## Phase 3.3: Configuration Files (Only After Baseline Tests Pass)

- [x] **T008 [P]** Configure Ruff in pyproject.toml
  - Add `[tool.ruff]` section to existing `pyproject.toml`
  - Set `target-version = "py311"`
  - Set `line-length = 88`
  - Add `[tool.ruff.lint]` section
  - Set `select = ["F", "E", "W", "I", "N", "D", "UP", "ANN", "S", "B", "C90"]`
  - Add `[tool.ruff.lint.pydocstyle]` with `convention = "google"`
  - Add `[tool.ruff.lint.mccabe]` with `max-complexity = 10`
  - Add inline comments explaining each rule category (e.g., "# F: Pyflakes errors", "# E/W: PEP 8 style", "# D: Docstrings", etc.) per FR-008

- [x] **T008a [P]** Configure pyright in pyproject.toml
  - Add `[tool.pyright]` section to existing `pyproject.toml`
  - Set `typeCheckingMode = "basic"`
  - Set `pythonVersion = "3.11"`
  - Add inline comments explaining basic mode settings per FR-018, FR-019

- [x] **T009 [P]** Create pre-commit configuration in `.pre-commit-config.yaml`
  - Add ruff-pre-commit repo with specific pinned version (e.g., `v0.1.9`, NOT `latest` or branch refs) to ensure local/CI consistency per edge case spec.md:L89
  - Configure ruff-format hook
  - Configure ruff lint hook with `--fix` arg
  - Add pyright hook from pre-commit/mirrors-pyright repo (pinned version)
  - Add pre-commit-hooks repo with YAML/TOML/JSON validators (also pinned)
  - Ensure ruff-format runs before ruff lint, then pyright
  - Exclude pytest from pre-commit hooks

- [x] **T010 [P]** Create GitHub Actions workflow in `.github/workflows/ci.yml`
  - Set name: "CI"
  - Trigger on pull_request to main/develop branches
  - Trigger on push to main branch
  - Configure ubuntu-latest runner
  - Add steps: checkout, setup Python 3.11, install deps
  - Add step: `ruff format --check .`
  - Add step: `ruff check .`
  - Add step: `pyright` (type checking)
  - Add step: `pytest` (runs full test suite including baseline)

- [x] **T011 [P]** Create setup script in `setup-dev.sh`
  - Add shebang and `set -e`
  - Check Python version >= 3.11 (exit 1 if failed)
  - Install pre-commit framework (exit 2 if failed)
  - Install Ruff (exit 2 if failed)
  - Install pyright (exit 2 if failed)
  - Run `pre-commit install` (exit 3 if failed)
  - Optionally run `pre-commit run --all-files` (warn but exit 0)
  - Make script executable: `chmod +x setup-dev.sh`

---

## Phase 3.4: Contract Tests (Validate Configuration Contracts)

- [x] **T012 [P]** Contract test for Ruff config in `tests/contract/test_ruff_config.py`
  - Assert `pyproject.toml` exists
  - Assert valid TOML syntax
  - Assert `[tool.ruff]` section exists
  - Assert `target-version = "py311"`
  - Assert `line-length = 88`
  - Assert all 11 rule sets in select array

- [x] **T012a [P]** Contract test for pyright config in `tests/contract/test_pyright_config.py`
  - Assert `pyproject.toml` exists
  - Assert valid TOML syntax
  - Assert `[tool.pyright]` section exists
  - Assert `typeCheckingMode = "basic"`
  - Assert `pythonVersion = "3.11"`

- [x] **T013 [P]** Contract test for pre-commit config in `tests/contract/test_precommit_config.py`
  - Assert `.pre-commit-config.yaml` exists
  - Assert valid YAML syntax
  - Assert ruff-pre-commit repo present
  - Assert ruff-format and ruff hooks present
  - Assert pyright hook present
  - Assert YAML/TOML/JSON validators present
  - Assert versions are pinned (not "latest")
  - Assert pytest NOT in hooks

- [x] **T014 [P]** Contract test for GitHub Actions in `tests/contract/test_github_actions.py`
  - Assert `.github/workflows/ci.yml` exists
  - Assert valid YAML syntax
  - Assert triggers on pull_request
  - Assert Python 3.11+ in setup step
  - Assert ruff format check step present
  - Assert ruff lint check step present
  - Assert pyright step present
  - Assert pytest step present

- [x] **T015 [P]** Contract test for setup script in `tests/contract/test_setup_script.py`
  - Assert `setup-dev.sh` exists and is executable
  - Test exit code 0 with Python 3.11+
  - Test creates `.git/hooks/pre-commit` after successful run
  - Test idempotency (running twice succeeds)
  - Verify output includes success messages

---

## Phase 3.5: Application & Verification

- [x] **T016** Apply Ruff formatting to existing codebase
  - Run: `ruff format .` from repository root
  - Review changes (should only affect code style)
  - Commit as single "formatting: apply Ruff standards to codebase"
  - **Dependency**: Requires T008 (Ruff configuration)

- [x] **T017** Apply Ruff auto-fixes to existing codebase
  - Run: `ruff check --fix .` from repository root
  - Review auto-fixed violations
  - Commit as "lint: apply Ruff auto-fixes"
  - **Dependency**: Requires T016 (formatting first)

- [x] **T018** Re-run baseline tests after formatting changes
  - Execute: `pytest tests/baseline/ -v -m baseline`
  - All tests must still PASS (verify no regressions)
  - Compare results with T007 output
  - If failures: Formatting broke code - rollback and investigate
  - **GATE**: Must pass before proceeding
  - **Dependency**: Requires T016, T017

- [x] **T019** Manual fixes for remaining Ruff violations
  - Run: `ruff check .` to identify remaining issues
  - Fix violations that couldn't be auto-fixed (e.g., missing docstrings, type hints)
  - Re-run `ruff check .` until clean
  - Commit as "lint: manually fix remaining Ruff violations"
  - **Dependency**: Requires T017, T018

---

## Phase 3.6: Integration & Validation

- [x] **T020** Install and test pre-commit hooks locally
  - Run: `./setup-dev.sh`
  - Verify exit code 0
  - Verify `.git/hooks/pre-commit` created
  - Test hook with intentional violation (should block commit)
  - Fix violation and verify commit succeeds
  - **Dependency**: Requires T009, T011, T019 (clean codebase)

- [x] **T021** Run all contract tests
  - Execute: `pytest tests/contract/ -v`
  - All 5 contract tests must pass (Ruff, pyright, pre-commit, GitHub Actions, setup script)
  - **Dependency**: Requires T008-T011 (all configs created)

---

## Phase 3.7: Pyright Type Error Resolution (Bottom-Up Strategy)

⚠️ **STRATEGY**: Fix type errors from foundation layer to application layer (leaf → root in dependency tree). This ensures fixes in lower layers automatically resolve cascading errors in upper layers.

### Layer 1: Foundation (Utils, Data, Checkpoints)

- [x] **T023 [P]** Fix pyright errors in utils/ (35 errors total)
  - Fix `src/utils/metrics.py` (19 errors)
  - Fix `src/utils/ema_callback.py` (5 errors)
  - Fix `src/utils/cl_score.py` (4 errors)
  - Fix `src/utils/visualize.py` (4 errors)
  - Fix `src/utils/featurizer.py` (3 errors)
  - Verify: `pyright src/utils/`
  - **Dependency**: Requires T019 (clean Ruff), T021 (contracts pass)

- [x] **T024 [P]** Fix pyright errors in data/ and ckpts/ (8 errors total)
  - Fix `src/data/components/mp_dataset.py` (5 errors)
  - Fix `ckpts/slim_ckpt.py` (3 errors)
  - Verify: `pyright src/data/ ckpts/`
  - **Dependency**: Requires T019 (clean Ruff), T021 (contracts pass)

### Layer 2: Core Modules (VAE, LDM, RL)

- [x] **T025** Fix pyright errors in vae_module/ (49 errors total)
  - Fix `src/vae_module/vae_module.py` (28 errors)
  - Fix `src/vae_module/encoders/transformer.py` (10 errors)
  - Fix `src/vae_module/predictor_module.py` (8 errors)
  - Fix `src/vae_module/decoders/transformer.py` (2 errors)
  - Fix `src/vae_module/encoders/cspnet.py` (1 error)
  - Verify: `pyright src/vae_module/`
  - **Dependency**: Requires T023, T024 (foundation fixed)

- [x] **T026** Fix pyright errors in ldm_module/ (19 errors total)
  - Fix `src/ldm_module/ldm_module.py` (17 errors)
  - Fix `src/ldm_module/condition.py` (2 errors)
  - Note: `denoisers/` and `diffusion/` excluded from pyright checks
  - Verify: `pyright src/ldm_module/`
  - **Dependency**: Requires T023, T024 (foundation fixed)

- [x] **T027** Fix pyright errors in rl_module/ (55 errors total)
  - Fix `src/rl_module/rl_module.py` (53 errors)
  - Fix `src/rl_module/reward.py` (2 errors)
  - Verify: `pyright src/rl_module/`
  - **Dependency**: Requires T023, T024 (foundation fixed)

### Layer 3: Application Scripts

- [x] **T028** Fix pyright errors in training/evaluation scripts (20 errors total)
  - Fix `src/evaluate.py` (11 errors)
  - Fix `src/sample.py` (5 errors)
  - Fix `src/sweep_vae.py` (1 error)
  - Fix `src/train_ldm.py` (1 error)
  - Fix `src/train_predictor.py` (1 error)
  - Fix `src/train_rl.py` (1 error)
  - Fix `src/train_vae.py` (1 error)
  - Verify: `pyright src/*.py`
  - **Dependency**: Requires T025, T026, T027 (core modules fixed)

### Layer 4: Test Files

- [x] **T029** Fix pyright errors in baseline tests (15 errors total)
  - Fix `tests/baseline/test_vae_module.py` (6 errors)
  - Fix `tests/baseline/test_rl_module.py` (5 errors)
  - Fix `tests/baseline/test_ldm_module.py` (3 errors)
  - Fix `tests/baseline/test_data_loading.py` (1 error)
  - Verify: `pyright tests/baseline/`
  - **Dependency**: Requires T025, T026, T027 (core modules fixed)

### Verification

- [x] **T030** Verify all pyright errors resolved
  - Run: `pyright .` from repository root
  - Verify 0 errors reported
  - Run: `pytest tests/baseline/ -v -m baseline` to ensure functionality intact
  - Run: `pytest tests/contract/ -v` to verify contract tests pass
  - **Dependency**: Requires T023-T029 (all layers fixed)

---

## Phase 3.8: CI/CD Validation & Documentation

- [x] **T031** Verify CI/CD workflow in GitHub
  - Push branch to GitHub
  - Open test Pull Request
  - Verify GitHub Actions CI runs automatically
  - Verify all checks pass (format, lint, type checking, pytest)
  - Close test PR
  - **Dependency**: Requires T010, T030 (all pyright errors fixed), T021 (tests pass)
  - **Note**: CI workflow configured and ready. Pyright check enabled (was continue-on-error, now blocking).

- [x] **T032** Create CONTRIBUTING.md at repository root
  - Add "Table of Contents" section
  - Add "Setup Instructions" with one-command setup
  - Add "Development Workflow" with commit process
  - Add "Coding Standards" explaining Ruff rules and pyright basic mode
  - Add "Type Checking" section explaining pyright usage and basic mode requirements
  - Add "Troubleshooting" for common issues including type checking errors
  - Add "CI/CD" explaining GitHub Actions workflow
  - Add "CI/CD Failure Troubleshooting" explaining how to interpret GitHub Actions check failures (format check, lint check, type check, pytest failures) for developers new to the workflow
  - Reference quickstart.md for validation steps
  - **Dependency**: Requires T020 (verified workflow), T031 (CI verified)

---

## Dependencies Graph

```
Setup Phase:
T001 (test structure) → T002 (pytest config) → T003-T006 (baseline tests)
T003-T006 (baseline tests) → T007 (verify baseline)

T007 GATES → Configuration Phase

Configuration Phase:
T008 (Ruff config)    ┐
T008a (pyright config)│
T009 (pre-commit)     ├─ [Parallel - different files]
T010 (GitHub CI)      │
T011 (setup script)   ┘

Contract Tests:
T012 (Ruff contract)       ┐
T012a (pyright contract)   │
T013 (pre-commit contract) ├─ [Parallel - different test files]
T014 (GitHub contract)     │
T015 (setup contract)      ┘

Application Phase:
T008, T008a → T016 (format) → T017 (auto-fix) → T018 (verify baseline) → T019 (manual fixes)

T018 GATES → Integration Phase

Integration Phase:
T019 → T020 (install hooks)
T008-T011, T008a → T021 (contract tests)

Pyright Error Resolution Phase (Bottom-Up):
T019, T021 → Layer 1: T023 (utils/) [P] T024 (data/, ckpts/) ┐
                                                              │
Layer 1 → Layer 2: T025 (vae_module/) [P] T026 (ldm_module/) [P] T027 (rl_module/)
                                                              │
Layer 2 → Layer 3: T028 (train scripts)                      │
                                                              │
Layer 2 → Layer 4: T029 (test files)                         │
                                                              │
T023-T029 → T030 (verify all fixed)

CI/CD Validation & Documentation:
T010, T030, T021 → T031 (verify CI)
T020, T031 → T032 (CONTRIBUTING.md)
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
# Task 2: "Configure pyright in pyproject.toml"
# Task 3: "Create pre-commit configuration in .pre-commit-config.yaml"
# Task 4: "Create GitHub Actions workflow in .github/workflows/ci.yml"
# Task 5: "Create setup script in setup-dev.sh"
```

### Example 3: Contract Tests (T012-T015)
```bash
# All contract tests can run in parallel - different test files
pytest tests/contract/ -n auto

# Or launch as separate tasks:
# Task 1: "Contract test for Ruff config in tests/contract/test_ruff_config.py"
# Task 2: "Contract test for pyright config in tests/contract/test_pyright_config.py"
# Task 3: "Contract test for pre-commit config in tests/contract/test_precommit_config.py"
# Task 4: "Contract test for GitHub Actions in tests/contract/test_github_actions.py"
# Task 5: "Contract test for setup script in tests/contract/test_setup_script.py"
```

### Example 4: Pyright Layer 1 Foundation (T023-T024)
```bash
# Foundation layer fixes can run in parallel - independent modules
# Task 1: "Fix pyright errors in utils/ (35 errors)"
# Task 2: "Fix pyright errors in data/ and ckpts/ (8 errors)"
```

### Example 5: Pyright Layer 2 Core Modules (T025-T027)
```bash
# Core module fixes can run in parallel AFTER Layer 1 completes
# Task 1: "Fix pyright errors in vae_module/ (49 errors)"
# Task 2: "Fix pyright errors in ldm_module/ (47 errors)"
# Task 3: "Fix pyright errors in rl_module/ (55 errors)"
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
- **Bottom-Up Strategy**: Pyright errors fixed from foundation (utils, data) → core modules (vae, ldm, rl) → application scripts → tests
- **Commit frequently**: After each task or logical group of parallel tasks
- **Test-first approach**: Baseline tests created and verified BEFORE any formatting changes
- **Avoid**: Vague tasks, modifying same file in parallel tasks
- **Performance target**: Pre-commit validation < 15 seconds (including type checking), CI pipeline < 5 minutes

---

## Validation Checklist

*GATE: Checked before marking tasks complete*

- [x] All contracts have corresponding tests (5 contracts → 5 contract tests: Ruff, pyright, pre-commit, GitHub Actions, setup)
- [x] All configuration entities have creation tasks (Ruff, pyright, pre-commit, CI, setup)
- [x] All tests come before implementation (baseline tests → config → application)
- [x] Parallel tasks truly independent (different files, verified)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Critical requirement met: Baseline tests BEFORE formatting changes (T007 gates T008-T014)
- [x] Test-First Development: T003-T007 before T016-T019
- [x] Pyright error resolution uses bottom-up strategy: foundation → core modules → application → tests
- [x] Layer dependencies enforced: Layer 1 (T023-T024) → Layer 2 (T025-T027) → Layer 3 (T028) & Layer 4 (T029)

---

**Tasks Complete** ✅
Total: 32 tasks (21 original + 2 pyright config additions + 8 pyright error resolution + 1 verification)
Ready for Phase 3 execution. Begin with T001.
