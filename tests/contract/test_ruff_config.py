"""Contract test for Ruff configuration.

This test validates that pyproject.toml contains a valid Ruff configuration
with all required settings matching the project requirements.
"""

import tomllib
from pathlib import Path

import pytest


@pytest.fixture
def pyproject_toml_path():
    """Return the path to pyproject.toml at repository root."""
    return Path(__file__).parent.parent.parent / "pyproject.toml"


@pytest.fixture
def pyproject_data(pyproject_toml_path):
    """Load and parse pyproject.toml."""
    with open(pyproject_toml_path, "rb") as f:
        return tomllib.load(f)


def test_pyproject_toml_exists(pyproject_toml_path) -> None:
    """Test that pyproject.toml exists at repository root."""
    assert pyproject_toml_path.exists(), "pyproject.toml not found"


def test_pyproject_toml_valid_syntax(pyproject_data) -> None:
    """Test that pyproject.toml has valid TOML syntax."""
    assert pyproject_data is not None
    assert isinstance(pyproject_data, dict)


def test_ruff_section_exists(pyproject_data) -> None:
    """Test that [tool.ruff] section exists."""
    assert "tool" in pyproject_data, "[tool] section missing"
    assert "ruff" in pyproject_data["tool"], "[tool.ruff] section missing"


def test_ruff_target_version(pyproject_data) -> None:
    """Test that target-version is set to py311."""
    ruff_config = pyproject_data["tool"]["ruff"]
    assert "target-version" in ruff_config, "target-version not configured"
    assert ruff_config["target-version"] == "py311", (
        f"Expected target-version='py311', got '{ruff_config['target-version']}'"
    )


def test_ruff_line_length(pyproject_data) -> None:
    """Test that line-length is set to 88 (Black-compatible)."""
    ruff_config = pyproject_data["tool"]["ruff"]
    assert "line-length" in ruff_config, "line-length not configured"
    assert ruff_config["line-length"] == 88, (
        f"Expected line-length=88, got {ruff_config['line-length']}"
    )


def test_ruff_lint_section_exists(pyproject_data) -> None:
    """Test that [tool.ruff.lint] section exists."""
    ruff_config = pyproject_data["tool"]["ruff"]
    assert "lint" in ruff_config, "[tool.ruff.lint] section missing"


def test_ruff_lint_select_rules(pyproject_data) -> None:
    """Test that all required rule sets are enabled in select array."""
    ruff_lint = pyproject_data["tool"]["ruff"]["lint"]
    assert "select" in ruff_lint, "select array not configured"

    select_rules = ruff_lint["select"]
    assert isinstance(select_rules, list), "select should be a list"

    # All 11 required rule sets as per spec
    required_rules = ["F", "E", "W", "I", "N", "D", "UP", "ANN", "S", "B", "C90"]

    for rule in required_rules:
        assert rule in select_rules, f"Required rule '{rule}' missing from select array"


def test_ruff_pydocstyle_convention(pyproject_data) -> None:
    """Test that pydocstyle convention is set to 'google'."""
    ruff_lint = pyproject_data["tool"]["ruff"]["lint"]
    assert "pydocstyle" in ruff_lint, "[tool.ruff.lint.pydocstyle] section missing"

    pydocstyle_config = ruff_lint["pydocstyle"]
    assert "convention" in pydocstyle_config, "convention not configured"
    assert pydocstyle_config["convention"] == "google", (
        f"Expected convention='google', got '{pydocstyle_config['convention']}'"
    )


def test_ruff_mccabe_complexity(pyproject_data) -> None:
    """Test that mccabe max-complexity is set to 10."""
    ruff_lint = pyproject_data["tool"]["ruff"]["lint"]
    assert "mccabe" in ruff_lint, "[tool.ruff.lint.mccabe] section missing"

    mccabe_config = ruff_lint["mccabe"]
    assert "max-complexity" in mccabe_config, "max-complexity not configured"
    assert mccabe_config["max-complexity"] == 10, (
        f"Expected max-complexity=10, got {mccabe_config['max-complexity']}"
    )
