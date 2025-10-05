"""Contract test for pre-commit configuration.

This test validates that .pre-commit-config.yaml exists and contains
all required hooks with proper configuration.
"""

import re
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def precommit_config_path():
    """Return the path to .pre-commit-config.yaml at repository root."""
    return Path(__file__).parent.parent.parent / ".pre-commit-config.yaml"


@pytest.fixture
def precommit_data(precommit_config_path):
    """Load and parse .pre-commit-config.yaml."""
    with open(precommit_config_path) as f:
        return yaml.safe_load(f)


def test_precommit_config_exists(precommit_config_path) -> None:
    """Test that .pre-commit-config.yaml exists at repository root."""
    assert precommit_config_path.exists(), ".pre-commit-config.yaml not found"


def test_precommit_config_valid_yaml(precommit_data) -> None:
    """Test that .pre-commit-config.yaml has valid YAML syntax."""
    assert precommit_data is not None
    assert isinstance(precommit_data, dict)


def test_precommit_repos_exists(precommit_data) -> None:
    """Test that repos key exists and is a list."""
    assert "repos" in precommit_data, "repos key missing"
    assert isinstance(precommit_data["repos"], list), "repos should be a list"
    assert len(precommit_data["repos"]) > 0, "repos list is empty"


def test_ruff_precommit_repo_present(precommit_data) -> None:
    """Test that ruff-pre-commit repo is present."""
    repos = precommit_data["repos"]
    ruff_repos = [r for r in repos if "astral-sh/ruff-pre-commit" in r.get("repo", "")]
    assert len(ruff_repos) > 0, "ruff-pre-commit repo not found"


def test_ruff_hooks_present(precommit_data) -> None:
    """Test that ruff-format and ruff hooks are present."""
    repos = precommit_data["repos"]
    ruff_repos = [r for r in repos if "astral-sh/ruff-pre-commit" in r.get("repo", "")]
    assert len(ruff_repos) > 0, "ruff-pre-commit repo not found"

    ruff_repo = ruff_repos[0]
    hooks = ruff_repo.get("hooks", [])
    hook_ids = [h.get("id") for h in hooks]

    assert "ruff-format" in hook_ids, "ruff-format hook not found"
    assert "ruff" in hook_ids, "ruff hook not found"


def test_pyright_hook_present(precommit_data) -> None:
    """Test that pyright hook is present."""
    repos = precommit_data["repos"]
    pyright_repos = [r for r in repos if "pyright" in r.get("repo", "").lower()]
    assert len(pyright_repos) > 0, "pyright repo not found"

    pyright_repo = pyright_repos[0]
    hooks = pyright_repo.get("hooks", [])
    hook_ids = [h.get("id") for h in hooks]

    assert "pyright" in hook_ids, "pyright hook not found"


def test_file_validators_present(precommit_data) -> None:
    """Test that YAML/TOML/JSON validators are present."""
    repos = precommit_data["repos"]
    precommit_hooks_repos = [
        r for r in repos if "pre-commit/pre-commit-hooks" in r.get("repo", "")
    ]
    assert len(precommit_hooks_repos) > 0, "pre-commit-hooks repo not found"

    hooks_repo = precommit_hooks_repos[0]
    hooks = hooks_repo.get("hooks", [])
    hook_ids = [h.get("id") for h in hooks]

    assert "check-yaml" in hook_ids, "check-yaml hook not found"
    assert "check-toml" in hook_ids, "check-toml hook not found"
    assert "check-json" in hook_ids, "check-json hook not found"


def test_versions_are_pinned(precommit_data) -> None:
    """Test that versions are pinned (not 'latest' or branch refs)."""
    repos = precommit_data["repos"]

    for repo in repos:
        rev = repo.get("rev", "")
        assert rev, f"No rev specified for repo {repo.get('repo')}"

        # Check that it's not 'latest' or a branch name
        assert rev != "latest", (
            f"Repo {repo.get('repo')} uses 'latest' - should be pinned"
        )
        assert not rev.startswith("HEAD"), (
            f"Repo {repo.get('repo')} uses HEAD ref - should be pinned"
        )
        assert not rev.startswith("main"), (
            f"Repo {repo.get('repo')} uses main branch - should be pinned"
        )
        assert not rev.startswith("master"), (
            f"Repo {repo.get('repo')} uses master branch - should be pinned"
        )

        # Valid pinned versions should start with 'v' or be a commit hash
        assert rev.startswith("v") or re.match(r"^[0-9a-f]{7,40}$", rev), (
            f"Repo {repo.get('repo')} has invalid version format: {rev}"
        )


def test_pytest_not_in_hooks(precommit_data) -> None:
    """Test that pytest is NOT in pre-commit hooks (should run in CI only)."""
    repos = precommit_data["repos"]

    for repo in repos:
        hooks = repo.get("hooks", [])
        for hook in hooks:
            hook_id = hook.get("id", "")
            assert "pytest" not in hook_id.lower(), (
                "pytest found in pre-commit hooks - should be CI only"
            )
