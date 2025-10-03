<!--
Sync Impact Report (Constitution Update)
Version: 1.0.0 → 2.0.0
Rationale: Initial constitution establishment for Chameleon2 project (MAJOR bump)

Modified Principles:
- All principles established from template (new baseline)

Added Sections:
- Core Principles (5 principles)
- Testing & Quality Standards
- Development Workflow
- Governance

Removed Sections: None

Templates Requiring Updates:
- ✅ .specify/templates/plan-template.md - Reviewed, constitution checks align
- ✅ .specify/templates/spec-template.md - Reviewed, scope alignment maintained
- ✅ .specify/templates/tasks-template.md - Reviewed, task categorization aligns
- ✅ .specify/templates/agent-file-template.md - Reviewed, no agent-specific references needed

Follow-up TODOs: None
-->

# Chameleon2 Constitution

## Core Principles

### I. Research Code Quality
Machine learning research code MUST maintain production-level quality standards. Code clarity and maintainability are NON-NEGOTIABLE, even for experimental features. All code MUST follow consistent formatting (Ruff), pass linting checks, and include docstrings for public APIs.

**Rationale**: Research code often becomes production code. Poor quality accumulates technical debt that blocks reproducibility, collaboration, and publication. The cost of maintaining quality from the start is far lower than retroactive cleanup.

### II. Configuration-Driven Design
Model architectures, hyperparameters, and training settings MUST be externalized to configuration files (Hydra). Hard-coded values in source files are PROHIBITED except for fundamental constants (e.g., physical constants, mathematical definitions).

**Rationale**: ML experiments require rapid iteration over parameters. Configuration-driven design enables experiment tracking, reproducibility, and systematic hyperparameter exploration without code modification.

### III. Modular Architecture
The project MUST maintain clear separation between VAE, LDM, and RL modules. Each module MUST be independently testable and reusable. Cross-module dependencies MUST flow through well-defined interfaces (PyTorch Lightning modules).

**Rationale**: Crystal structure generation involves distinct learning stages. Modularity enables independent development, testing, and improvement of each stage without cascading changes.

### IV. Data Integrity & Reproducibility
All experiments MUST log random seeds, data splits, and configuration snapshots. Data preprocessing MUST be deterministic and versioned. Results MUST be reproducible from logged configurations.

**Rationale**: Scientific validity requires reproducibility. ML experiments have multiple sources of randomness (initialization, data shuffling, stochastic training). Comprehensive logging is the only defense against irreproducible results.

### V. Test-Driven Development (NON-NEGOTIABLE)
All new features MUST have tests written BEFORE implementation. Baseline tests MUST validate existing functionality before refactoring. Tests MUST fail initially (Red-Green-Refactor cycle). CI/CD pipelines MUST enforce comprehensive test execution before merge. Pre-commit hooks SHOULD focus on fast feedback (formatting, linting) to maintain developer velocity, with test execution reserved for CI/CD.

**Rationale**: ML code is particularly prone to silent failures (e.g., incorrect tensor shapes, gradient bugs, metric calculation errors). Tests catch these issues early. The TDD cycle ensures tests actually validate the intended behavior. Pre-commit hooks provide immediate feedback on code style, while CI/CD gates ensure no untested code reaches the main branch.

## Testing & Quality Standards

### Test Coverage Requirements
- **Baseline Tests**: Smoke tests for core modules (VAE, LDM, RL) MUST pass before any major refactoring
- **Integration Tests**: End-to-end training pipelines MUST be validated with small-scale tests
- **Unit Tests**: Critical components (loss functions, metrics, data transforms) MUST have unit tests
- **Performance Tests**: Training time and memory usage benchmarks SHOULD be tracked

### Code Quality Gates
- **Formatting**: Ruff format MUST pass (enforced by pre-commit hooks)
- **Linting**: Ruff linting rules MUST pass (strict configuration)
- **Type Hints**: Public APIs SHOULD include type hints (gradual adoption)
- **Documentation**: Module and class docstrings MUST be present

## Development Workflow

### Branching & Commits
- Feature development MUST use descriptive branch names (e.g., `001-feature-name`)
- Commits MUST be atomic and include meaningful messages
- Pre-commit hooks MUST pass before commits are accepted
- Breaking changes MUST be clearly documented in commit messages

### Code Review Requirements
- All changes MUST pass CI/CD checks (GitHub Actions)
- Formatting and linting violations MUST be fixed before review
- Test failures block merging
- Configuration changes MUST include rationale in PR description

### Collaboration Standards
- New developers MUST complete one-command setup (documented in CONTRIBUTING.md)
- Merge conflicts from formatting changes MUST be coordinated with team
- Experimental code MUST still meet quality standards (no "temporary" exceptions)

## Governance

### Amendment Process
1. Proposed amendments MUST be documented with rationale
2. Breaking changes require MAJOR version bump
3. New principles require MINOR version bump
4. Clarifications require PATCH version bump

### Compliance & Enforcement
- All PRs MUST demonstrate constitution compliance
- CI/CD pipeline enforces automated checks (formatting, linting, tests)
- Manual review verifies adherence to non-automated principles
- Complexity that violates principles MUST be justified in implementation plans

### Documentation Requirements
- This constitution supersedes all conflicting practices
- The CLAUDE.md file provides runtime development guidance derived from this constitution
- Template files in `.specify/templates/` MUST align with these principles

**Version**: 2.0.1 | **Ratified**: 2025-10-03 | **Last Amended**: 2025-10-03
