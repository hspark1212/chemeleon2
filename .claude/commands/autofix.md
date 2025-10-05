---
description: Automatically fix pre-commit issues (ruff, pyright, pytest) and run validation
---

The user input can be provided directly by the agent or as a command argument - you **MUST** consider it before proceeding with the prompt (if not empty).

User input:

$ARGUMENTS

1. **Run pre-commit hooks to identify issues**:
   - Execute `pre-commit run --all-files` to see all current violations
   - Parse the output to identify which hooks failed (ruff-format, ruff, pyright, etc.)
   - Report the number and types of issues found

2. **Auto-fix Ruff issues**:
   - Run `ruff format .` to fix all formatting issues
   - Run `ruff check . --fix` to auto-fix linting violations
   - Re-run `pre-commit run --all-files` to verify ruff fixes
   - Report which ruff issues were fixed and which remain

3. **Fix Pyright type errors** (if pyright hook is enabled):
   - Run `pyright --outputjson` to get detailed type error information
   - For each type error:
     - Analyze the error message and context
     - Determine the appropriate fix (add type hints, fix type mismatches, etc.)
     - Apply the fix using Edit tool
     - Verify the fix resolves the error
   - Continue until all type errors are resolved or maximum iterations reached
   - Report pyright error resolution progress

4. **Fix pytest failures** (if user requests or if pytest fails):
   - Run `pytest` to identify failing tests
   - For each failing test:
     - Read the test file and implementation
     - Analyze the failure reason
     - Fix the underlying issue in implementation or update test if specification changed
     - Re-run the specific test to verify fix
   - Continue until all tests pass or maximum iterations reached
   - Report test fix progress

5. **Validation and verification**:
   - Run `pre-commit run --all-files` one final time
   - Run `pytest tests/` to ensure all tests pass
   - If all checks pass:
     - Report success with summary of all fixes applied
     - Suggest running `git add .` if changes should be staged
   - If issues remain:
     - Report remaining issues that couldn't be auto-fixed
     - Suggest manual intervention steps

6. **Safety and best practices**:
   - Always read files before editing to understand context
   - Make minimal changes to fix specific issues
   - Preserve existing code style and patterns
   - Run validation after each fix to prevent breaking changes
   - If a fix causes new errors, revert and try alternative approach

7. **Progress tracking**:
   - Use TodoWrite to track progress through each phase
   - Mark tasks as completed when each category is fixed
   - Provide clear status updates after each major step

Note: This command will attempt automatic fixes but may require user intervention for complex issues. Always verify fixes by running the validation suite.
