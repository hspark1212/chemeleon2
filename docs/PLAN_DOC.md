# Jupyter Book Documentation Plan for Chemeleon2

## Overview

Set up Jupyter Book documentation with GitHub Pages deployment for the Chemeleon2 project.

**User Requirements:**
- Use Jupyter Book as the documentation builder
- Deploy to GitHub Pages with auto-deploy on push to main
- Markdown docs only (no Jupyter notebooks)
- Include API Reference (auto-generated from docstrings)
- Include Architecture Guide (module explanations)

---

## Documentation Structure

```
docs/
├── _config.yml                  # Jupyter Book configuration
├── _toc.yml                     # Table of contents
├── index.md                     # Landing page
├── getting-started/
│   ├── installation.md          # Installation guide (from README)
│   └── quickstart.md            # Quick start guide
├── user-guide/
│   ├── training.md              # ← Migrated from TRAINING.md
│   ├── evaluation.md            # ← Migrated from EVALUATION.md
│   └── custom-rewards.md        # ← Migrated from TRAINING_CUSTOM_RL.md
├── architecture/
│   ├── index.md                 # Architecture overview with diagrams
│   ├── vae-module.md            # VAE module explanation
│   ├── ldm-module.md            # LDM module explanation
│   ├── rl-module.md             # RL module explanation
│   └── data-pipeline.md         # Data loading and utilities
├── api/
│   └── index.md                 # API reference (autodoc2 generated)
└── contributing.md              # ← Migrated from CONTRIBUTING.md
```

---

## Implementation Steps

### Step 1: Initialize Jupyter Book

Use `jupyter-book init` to scaffold the documentation structure:

```bash
# Remove existing docs temporarily or backup
mv docs docs_backup

# Initialize new Jupyter Book structure
mkdir -p docs && cd docs && source ../.venv/bin/activate && jupyter-book init

# This generates:
# - docs/_config.yml (base config)
# - docs/_toc.yml (base TOC)
# - docs/intro.md (template landing page)
# - docs/logo.png, docs/references.bib, etc.
```

Then customize the generated `_config.yml` with:
- Title: "Chemeleon2 Documentation"
- Disable notebook execution (`execute_notebooks: "off"`)
- Enable MyST extensions (admonitions, math, task lists)
- Configure autodoc2 for API generation from `src/` directory
- Set repository URL for edit buttons
- Configure baseurl for GitHub Pages

---

### Step 2: Customize Table of Contents

Edit `docs/_toc.yml` to define the structure:

```yaml
format: jb-book
root: index
parts:
  - caption: Getting Started
    chapters:
      - file: getting-started/installation
      - file: getting-started/quickstart
  - caption: User Guide
    chapters:
      - file: user-guide/training
      - file: user-guide/evaluation
      - file: user-guide/custom-rewards
  - caption: Architecture Guide
    chapters:
      - file: architecture/index
      - file: architecture/vae-module
      - file: architecture/ldm-module
      - file: architecture/rl-module
      - file: architecture/data-pipeline
  - caption: API Reference
    chapters:
      - file: api/index
  - caption: Development
    chapters:
      - file: contributing
```

---

### Step 3: Create New Content Files

| File | Description |
|------|-------------|
| `docs/index.md` | Landing page with overview, features, citation |
| `docs/getting-started/installation.md` | Detailed installation (expanded from README) |
| `docs/getting-started/quickstart.md` | Quick start tutorial |
| `docs/architecture/index.md` | Architecture overview with Mermaid diagrams |
| `docs/architecture/vae-module.md` | VAE module explanation |
| `docs/architecture/ldm-module.md` | LDM module explanation |
| `docs/architecture/rl-module.md` | RL module explanation |
| `docs/architecture/data-pipeline.md` | Data and utilities |
| `docs/api/index.md` | API reference landing page |

---

### Step 4: Migrate Existing Documentation

| Original | New Location | Action |
|----------|--------------|--------|
| `docs_backup/TRAINING.md` | `docs/user-guide/training.md` | Migrate & update links |
| `docs_backup/EVALUATION.md` | `docs/user-guide/evaluation.md` | Migrate & update links |
| `docs_backup/TRAINING_CUSTOM_RL.md` | `docs/user-guide/custom-rewards.md` | Migrate & update links |
| `docs_backup/CONTRIBUTING.md` | `docs/contributing.md` | Migrate & update links |

After migration, remove `docs_backup/` directory.

---

### Step 5: Create GitHub Actions Workflow

**File:** `.github/workflows/docs-deploy.yaml`

Workflow:
- **Trigger**: Push to main (paths: `docs/**`, `src/**/*.py`), manual dispatch
- **Build job**: Install docs deps → Build Jupyter Book
- **Deploy job**: Deploy to GitHub Pages (only on main branch push)

```yaml
name: Deploy Documentation

on:
  push:
    branches:
      - main
    paths:
      - 'docs/**'
      - 'src/**/*.py'
      - '.github/workflows/docs-deploy.yaml'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install jupyter-book sphinx-autodoc2 myst-nb sphinx-design sphinxcontrib-mermaid
      - name: Build documentation
        run: jupyter-book build docs
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs/_build/html

  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/deploy-pages@v4
        id: deployment
```

---

### Step 6: Update .gitignore

Add:
```gitignore
docs/_build/
docs/api/src/
```

---

### Step 7: Enable GitHub Pages

Manual step after first successful workflow:
1. Go to repository Settings → Pages
2. Set Source to "GitHub Actions"

---

## Critical Files to Create/Modify

| File | Action |
|------|--------|
| `docs/_config.yml` | **CUSTOMIZE** - After jupyter-book create |
| `docs/_toc.yml` | **CUSTOMIZE** - After jupyter-book create |
| `docs/index.md` | **CREATE** - Landing page |
| `docs/getting-started/installation.md` | **CREATE** |
| `docs/getting-started/quickstart.md` | **CREATE** |
| `docs/architecture/index.md` | **CREATE** |
| `docs/architecture/vae-module.md` | **CREATE** |
| `docs/architecture/ldm-module.md` | **CREATE** |
| `docs/architecture/rl-module.md` | **CREATE** |
| `docs/architecture/data-pipeline.md` | **CREATE** |
| `docs/api/index.md` | **CREATE** |
| `docs/user-guide/training.md` | **MIGRATE** from TRAINING.md |
| `docs/user-guide/evaluation.md` | **MIGRATE** from EVALUATION.md |
| `docs/user-guide/custom-rewards.md` | **MIGRATE** from TRAINING_CUSTOM_RL.md |
| `docs/contributing.md` | **MIGRATE** from CONTRIBUTING.md |
| `.github/workflows/docs-deploy.yaml` | **CREATE** - CI/CD workflow |
| `.gitignore` | Add build artifacts |

---

## Verification

After implementation:

```bash
# Build locally
jupyter-book build docs

# View at docs/_build/html/index.html
cd docs && jupyter-book build --html
```

---

## Summary

- **~16 files** to create/modify
- Use `jupyter-book create` to initialize structure
- **Auto-deploy** on push to main branch via `docs-deploy.yaml`
- **API docs** auto-generated via autodoc2
- **Architecture guide** with Mermaid diagrams
- Existing docs migrated to new structure

---

## TODO: Clean Up After Debugging

**⚠️ IMPORTANT: Remove these temporary changes after debugging is complete**

### 1. Remove feat branches from workflow trigger
**File:** `.github/workflows/docs-deploy.yaml`
- [ ] **Line 7**: Remove `- feat/custom-reward-001`
- [ ] **Line 8**: Remove `- feat/custom-reward-001-docs`

### 2. Restore deploy job condition
**File:** `.github/workflows/docs-deploy.yaml`
- [ ] **Line 53**: Uncomment `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`
- [ ] **Line 54-55**: Delete temporary deployment condition lines

**Final state (lines 51-56):**
```yaml
  deploy:
    # Only deploy on push to main (not on PRs)
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
```
