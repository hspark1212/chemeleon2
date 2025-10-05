
# Implementation Plan: Development Workflow Standards for Team Collaboration

**Branch**: `001-this-repository-was` | **Date**: 2025-10-03 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-this-repository-was/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Establish development workflow standards for team collaboration by implementing code formatting (Ruff Format), comprehensive linting (Ruff with strict rules), static type checking (pyright in basic mode), pre-commit hooks (blocking commits on failures), and CI/CD validation (GitHub Actions). **Critical priority**: Create baseline tests FIRST to verify code functionality before and after applying formatting changes, ensuring no regressions during retroactive code reformatting.

## Technical Context
**Language/Version**: Python 3.11+ (already specified in pyproject.toml)
**Primary Dependencies**: Ruff (formatter + linter), pyright (type checker in basic mode), pre-commit framework, pytest, GitHub Actions
**Storage**: Configuration files (.toml, .yaml), git repository
**Testing**: pytest (baseline tests before formatting, smoke tests for ML modules)
**Target Platform**: Linux development environments, GitHub CI/CD runners
**Project Type**: Single Python project (ML/scientific computing - VAE, LDM, RL modules)
**Performance Goals**: Pre-commit validation < 15 seconds (including type checking), CI/CD pipeline < 5 minutes
**Constraints**: Must not break existing functionality during retroactive formatting
**Scale/Scope**: ~45 Python files in src/, team collaboration workflow for 2+ developers
**User-Specified Priority**: Create baseline tests FIRST to validate code works before and after formatting changes

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Note**: Constitution file is a template placeholder. Applying standard software engineering principles:

✅ **Test-First Development**: Baseline tests created BEFORE formatting changes (user-specified requirement)
✅ **Non-Breaking Changes**: Formatting/linting are code style changes only, not functional changes
✅ **Automated Validation**: Pre-commit hooks and CI/CD ensure quality gates
✅ **Documentation**: CONTRIBUTING.md provides clear setup and usage guidance
✅ **Simplicity**: Using Ruff (single tool for format + lint) instead of multiple tools
✅ **Team Collaboration**: One-command setup for new developers

**Potential Concerns**:
- Retroactive formatting on existing code could cause merge conflicts → Mitigated by coordinating with team on timing
- No existing tests → Addressed by creating baseline tests FIRST before any formatting

**Status**: PASS - No constitutional violations. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Existing structure (ML/scientific Python project)
src/
├── vae_module/           # VAE models
├── ldm_module/           # Latent diffusion models
├── rl_module/            # Reinforcement learning
├── data/                 # Data loading and processing
├── utils/                # Utilities
└── *.py                  # Training scripts (train_vae.py, train_ldm.py, etc.)

# NEW: Test structure to be created (Phase 1)
tests/
├── baseline/             # Smoke tests to validate pre/post formatting
│   ├── test_vae_module.py
│   ├── test_ldm_module.py
│   ├── test_rl_module.py
│   └── test_data_loading.py
├── integration/          # Integration tests (future)
└── unit/                 # Unit tests (future)

# NEW: Configuration files (Phase 1)
pyproject.toml            # Ruff config (extend existing)
.pre-commit-config.yaml   # Pre-commit hooks
.github/
└── workflows/
    └── ci.yml            # CI/CD pipeline

# NEW: Documentation (Phase 1)
CONTRIBUTING.md           # Developer setup guide
```

**Structure Decision**: Single Python project (ML/scientific computing). Using existing `src/` structure with new `tests/` directory for baseline validation. Configuration files at repository root following Python community standards.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy** (Test-First Priority):
1. **Baseline Test Creation Tasks** (CRITICAL - Must come first)
   - Create `tests/` directory structure
   - Write baseline smoke tests for VAE, LDM, RL modules
   - Write overfit-single-batch test
   - Write data loading tests
   - Configure pytest in pyproject.toml
   - **Run tests and verify green** → Establishes functional baseline

2. **Configuration Tasks**
   - Configure Ruff in pyproject.toml
   - Create .pre-commit-config.yaml
   - Create GitHub Actions workflow
   - Create setup script (setup-dev.sh)

3. **Application Tasks** (Only after baseline tests pass)
   - Apply Ruff formatting to codebase (`ruff format .`)
   - Apply Ruff fixes (`ruff check --fix .`)
   - **Re-run baseline tests** → Verify no regressions
   - Manual fixes for remaining violations

4. **Documentation Tasks**
   - Create CONTRIBUTING.md
   - Update CLAUDE.md (via script)

**Ordering Strategy**:
- **TEST-FIRST**: Baseline tests BEFORE any formatting changes
- **Verification loops**: Test → Format → Test again
- **Dependency order**: Tests → Config → Application → Docs
- **Mark [P]** for parallel execution where safe (e.g., independent test files)

**Critical Success Factor**:
Tests must pass in step 1 and step 3. If step 3 tests fail, formatting broke something - must fix before proceeding.

**Estimated Output**: 15-20 numbered, strictly ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved (no unknowns in Technical Context)
- [x] Complexity deviations documented (none - follows TDD principle)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
