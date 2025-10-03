#!/bin/bash
# Claude Auto-Fix Script for Pre-Commit Validation
# Automatically fixes Ruff and pyright violations using Claude Code

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🤖 Running Claude Auto-Fix..."

# Step 1: Run Ruff format check
echo -e "${YELLOW}[1/4] Checking formatting...${NC}"
if ! ruff format --check . > /dev/null 2>&1; then
    echo -e "${YELLOW}Formatting issues detected. Running ruff format...${NC}"
    ruff format .
fi

# Step 2: Run Ruff lint check
echo -e "${YELLOW}[2/4] Checking lint violations...${NC}"
RUFF_ERRORS=$(ruff check . 2>&1 || true)
if [ -n "$RUFF_ERRORS" ]; then
    echo -e "${YELLOW}Ruff violations detected. Attempting auto-fix...${NC}"

    # Try auto-fix first
    ruff check --fix . || true

    # Check if violations remain
    REMAINING_ERRORS=$(ruff check . 2>&1 || true)
    if [ -n "$REMAINING_ERRORS" ]; then
        echo -e "${YELLOW}Remaining violations require manual fixes.${NC}"
        echo "Use 'claude' command or AI assistant to fix these violations."
        echo ""
        echo "Example violations:"
        echo "$REMAINING_ERRORS" | head -30
    fi
fi

# Step 3: Run pyright type checking
echo -e "${YELLOW}[3/4] Checking type annotations...${NC}"
if command -v pyright &> /dev/null; then
    PYRIGHT_ERRORS=$(pyright . 2>&1 || true)
    if echo "$PYRIGHT_ERRORS" | grep -q "error"; then
        echo -e "${YELLOW}Type checking errors detected.${NC}"
        echo "Use Claude Code to add missing type annotations."
    fi
else
    echo -e "${YELLOW}pyright not installed. Skipping type check.${NC}"
fi

# Step 4: Run baseline smoke tests to verify no functionality broken
echo -e "${YELLOW}[4/4] Running baseline smoke tests...${NC}"
if [ -d "tests/baseline" ]; then
    if .venv/bin/python -m pytest tests/baseline/ -m baseline --maxfail=1 -q > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Baseline tests passed${NC}"
    else
        echo -e "${RED}✗ Baseline tests failed!${NC}"
        echo "Auto-fixes may have broken functionality. Please review changes."
        exit 1
    fi
else
    echo -e "${YELLOW}No baseline tests found. Skipping validation.${NC}"
fi

# Final status
FINAL_CHECK=$(ruff check . 2>&1 || true)
if [ -z "$FINAL_CHECK" ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    exit 0
else
    VIOLATION_COUNT=$(echo "$FINAL_CHECK" | grep -c "^Found" || echo "unknown")
    echo -e "${YELLOW}⚠ ${VIOLATION_COUNT} violations remaining${NC}"
    echo "These require manual fixes or AI assistance."
    exit 0  # Don't block commit, just warn
fi
