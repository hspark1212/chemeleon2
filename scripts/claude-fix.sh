#!/bin/bash
# Claude Auto-Fix Wrapper Script
# User-friendly wrapper for Claude Code auto-fix functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Claude Code Auto-Fix Tool           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if Claude Code CLI is available
if ! command -v claude &> /dev/null; then
    echo -e "${YELLOW}⚠ Claude Code CLI not found${NC}"
    echo "This script works best with Claude Code installed."
    echo "Continuing with standard Ruff auto-fix only..."
    echo ""
fi

# Check if in git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo -e "${RED}✗ Not in a git repository${NC}"
    exit 1
fi

# Store current state
BEFORE_DIFF=$(git diff --stat 2>/dev/null || echo "No changes")

echo -e "${YELLOW}Running pre-commit checks...${NC}"

# Run pre-commit hooks
if pre-commit run --all-files 2>&1 | tee /tmp/precommit-output.txt; then
    echo -e "${GREEN}✓ All pre-commit checks passed!${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Pre-commit checks failed. Attempting auto-fix...${NC}"
echo ""

# Run Claude auto-fix script
if [ -x ".git-hooks/claude-autofix.sh" ]; then
    .git-hooks/claude-autofix.sh
else
    echo -e "${RED}✗ Claude auto-fix script not found or not executable${NC}"
    echo "Expected: .git-hooks/claude-autofix.sh"
    exit 1
fi

# Check if changes were made
AFTER_DIFF=$(git diff --stat 2>/dev/null || echo "No changes")

if [ "$BEFORE_DIFF" != "$AFTER_DIFF" ]; then
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Changes made by auto-fix:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    git diff --stat
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Show detailed diff
    echo -e "${YELLOW}Review changes? [y/N]${NC} "
    read -r REVIEW
    if [[ $REVIEW =~ ^[Yy]$ ]]; then
        git diff
    fi

    echo ""
    echo -e "${YELLOW}Accept these changes? [Y/n]${NC} "
    read -r ACCEPT
    if [[ ! $ACCEPT =~ ^[Nn]$ ]]; then
        echo -e "${GREEN}✓ Changes accepted${NC}"
        echo "You can now stage and commit these changes:"
        echo "  git add -u"
        echo "  git commit -m 'lint: fix Ruff violations'"
    else
        echo -e "${YELLOW}Reverting changes...${NC}"
        git restore .
        echo -e "${YELLOW}Changes reverted${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}No automatic fixes were applied${NC}"
    echo "You may need to manually fix the remaining violations."
fi

# Run tests if changes were made
if [ "$BEFORE_DIFF" != "$AFTER_DIFF" ] && [ -d "tests/baseline" ]; then
    echo ""
    echo -e "${YELLOW}Running baseline tests to verify fixes...${NC}"
    if .venv/bin/python -m pytest tests/baseline/ -m baseline -q; then
        echo -e "${GREEN}✓ All baseline tests passed${NC}"
    else
        echo -e "${RED}✗ Tests failed after auto-fix${NC}"
        echo -e "${YELLOW}Consider reverting changes with: git restore .${NC}"
        exit 1
    fi
fi

exit 0
