"""Contract test for GitHub Actions CI workflow.

This test validates that .github/workflows/ci.yml exists and contains
all required steps for continuous integration.
"""

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def ci_workflow_path():
    """Return the path to .github/workflows/ci.yml."""
    return Path(__file__).parent.parent.parent / ".github" / "workflows" / "ci.yml"


@pytest.fixture
def ci_workflow_data(ci_workflow_path):
    """Load and parse .github/workflows/ci.yml."""
    with open(ci_workflow_path) as f:
        return yaml.safe_load(f)


def test_ci_workflow_exists(ci_workflow_path) -> None:
    """Test that .github/workflows/ci.yml exists."""
    assert ci_workflow_path.exists(), ".github/workflows/ci.yml not found"


def test_ci_workflow_valid_yaml(ci_workflow_data) -> None:
    """Test that .github/workflows/ci.yml has valid YAML syntax."""
    assert ci_workflow_data is not None
    assert isinstance(ci_workflow_data, dict)


def test_workflow_name(ci_workflow_data) -> None:
    """Test that workflow has a name."""
    assert "name" in ci_workflow_data, "Workflow name missing"
    assert ci_workflow_data["name"], "Workflow name is empty"


def test_triggers_on_pull_request(ci_workflow_data) -> None:
    """Test that workflow triggers on pull_request events."""
    # YAML parses 'on' as True (boolean), so we need to check for True key
    assert True in ci_workflow_data, "Trigger configuration missing"
    triggers = ci_workflow_data[True]

    assert "pull_request" in triggers, "pull_request trigger missing"


def test_pull_request_branches(ci_workflow_data) -> None:
    """Test that pull_request trigger includes main and/or develop branches."""
    triggers = ci_workflow_data[True]
    pr_config = triggers.get("pull_request", {})

    # Can be a dict with branches or a simple trigger
    if isinstance(pr_config, dict) and "branches" in pr_config:
        branches = pr_config["branches"]
        assert isinstance(branches, list), "branches should be a list"
        # At least one of main/develop should be present
        assert any(b in branches for b in ["main", "develop"]), (
            "Neither 'main' nor 'develop' in pull_request branches"
        )


def test_triggers_on_push(ci_workflow_data) -> None:
    """Test that workflow triggers on push to main branch."""
    triggers = ci_workflow_data[True]
    assert "push" in triggers, "push trigger missing"


def test_jobs_exist(ci_workflow_data) -> None:
    """Test that jobs are defined."""
    assert "jobs" in ci_workflow_data, "jobs section missing"
    jobs = ci_workflow_data["jobs"]
    assert isinstance(jobs, dict), "jobs should be a dictionary"
    assert len(jobs) > 0, "No jobs defined"


def test_python_version_in_setup(ci_workflow_data) -> None:
    """Test that Python 3.11+ is configured in setup step."""
    jobs = ci_workflow_data["jobs"]

    # Find the first job
    job_name = list(jobs.keys())[0]
    job = jobs[job_name]

    steps = job.get("steps", [])
    assert len(steps) > 0, "No steps defined in job"

    # Find Python setup step
    python_setup_steps = [
        s for s in steps if "setup-python" in s.get("uses", "").lower()
    ]
    assert len(python_setup_steps) > 0, "Python setup step not found"

    python_step = python_setup_steps[0]
    with_config = python_step.get("with", {})
    python_version = with_config.get("python-version", "")

    # Check that Python 3.11 is specified (could be '3.11' or '3.11.x' or '>= 3.11')
    assert "3.11" in str(python_version), (
        f"Python 3.11 not found in version spec: {python_version}"
    )


def test_ruff_format_check_step(ci_workflow_data) -> None:
    """Test that ruff format check step is present."""
    jobs = ci_workflow_data["jobs"]
    job_name = list(jobs.keys())[0]
    job = jobs[job_name]
    steps = job.get("steps", [])

    # Find ruff format step
    ruff_format_steps = [
        s for s in steps if "run" in s and "ruff format" in s.get("run", "").lower()
    ]
    assert len(ruff_format_steps) > 0, "Ruff format check step not found"

    # Verify it uses --check flag
    ruff_format_step = ruff_format_steps[0]
    assert "--check" in ruff_format_step["run"], (
        "Ruff format step should use --check flag"
    )


def test_ruff_lint_check_step(ci_workflow_data) -> None:
    """Test that ruff lint check step is present."""
    jobs = ci_workflow_data["jobs"]
    job_name = list(jobs.keys())[0]
    job = jobs[job_name]
    steps = job.get("steps", [])

    # Find ruff lint step (should have 'ruff check' but not 'ruff format')
    ruff_lint_steps = [
        s
        for s in steps
        if "run" in s
        and "ruff check" in s.get("run", "").lower()
        and "format" not in s.get("run", "").lower()
    ]
    assert len(ruff_lint_steps) > 0, "Ruff lint check step not found"


def test_pyright_step(ci_workflow_data) -> None:
    """Test that pyright type checking step is present."""
    jobs = ci_workflow_data["jobs"]
    job_name = list(jobs.keys())[0]
    job = jobs[job_name]
    steps = job.get("steps", [])

    # Find pyright step
    pyright_steps = [
        s for s in steps if "run" in s and "pyright" in s.get("run", "").lower()
    ]
    assert len(pyright_steps) > 0, "Pyright type check step not found"


def test_pytest_step(ci_workflow_data) -> None:
    """Test that pytest step is present."""
    jobs = ci_workflow_data["jobs"]
    job_name = list(jobs.keys())[0]
    job = jobs[job_name]
    steps = job.get("steps", [])

    # Find pytest step
    pytest_steps = [
        s for s in steps if "run" in s and "pytest" in s.get("run", "").lower()
    ]
    assert len(pytest_steps) > 0, "pytest step not found"
