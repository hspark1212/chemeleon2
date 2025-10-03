# Claude Code Pre-Commit Auto-Fix Hook

This is a configuration file for Claude Code to automatically fix Ruff and pyright violations.

## Purpose

When developers run into Ruff linting or pyright type checking errors, this hook provides Claude Code with instructions on how to fix common violations automatically.

## Instructions for Claude Code

When fixing Ruff violations, follow these guidelines:

### 1. Missing Docstrings (D-series rules)

**For modules (D100)**:
```python
"""Module description.

This module provides [brief description of purpose].
"""
```

**For classes (D101)**:
```python
class MyClass:
    """Brief description of the class.

    Longer description if needed explaining purpose, usage, etc.
    """
```

**For methods (D102)**:
```python
def method_name(self, arg: str) -> int:
    """Brief description of what method does.

    Args:
        arg: Description of argument.

    Returns:
        Description of return value.
    """
```

**For functions (D103)**:
```python
def function_name(param: str) -> bool:
    """Brief description of function purpose.

    Args:
        param: Description of parameter.

    Returns:
        Description of return value.
    """
```

### 2. Missing Type Annotations (ANN-series rules)

**For function arguments (ANN001)**:
```python
# Before
def process(data, verbose=False):
    pass

# After
def process(data: list[str], verbose: bool = False) -> None:
    pass
```

**For return types (ANN201)**:
```python
# Before
def calculate(x: int, y: int):
    return x + y

# After
def calculate(x: int, y: int) -> int:
    return x + y
```

**For __init__ methods (ANN204)**:
```python
# Always return None
def __init__(self, name: str) -> None:
    self.name = name
```

### 3. Line Too Long (E501)

Break long lines using parentheses, backslashes, or string concatenation:

```python
# Before
some_very_long_function_call(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)

# After
some_very_long_function_call(
    arg1, arg2, arg3, arg4,
    arg5, arg6, arg7, arg8
)
```

### 4. Assert Statements in Production Code (S101)

**In tests**: Asserts are fine, ignore S101 in test files

**In production code**: Replace with proper error handling:
```python
# Before
assert value is not None

# After
if value is None:
    raise ValueError("Value must not be None")
```

### 5. Complexity Issues (C901)

Refactor complex functions by:
- Extracting helper functions
- Using early returns
- Simplifying conditional logic
- Breaking into smaller methods

### 6. Import Ordering (I001)

Follow PEP 8 import order:
1. Standard library imports
2. Third-party imports
3. Local application imports

Each group separated by blank line.

### 7. Naming Conventions (N-series)

- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

## Output Format

Apply fixes directly to the files. Do not provide explanations unless errors occur.

## Safety Rules

1. **Never break existing tests**: Always run baseline tests after fixes
2. **Preserve functionality**: Only change style/types, not logic
3. **Use Google-style docstrings**: Match existing codebase style
4. **Be conservative with types**: Use `Any` if uncertain, don't guess
5. **Ask before major refactoring**: Complexity fixes may need human review

## Example Usage

When Claude Code receives errors like:
```
src/example.py:10:5: D103 Missing docstring in public function
src/example.py:15:20: ANN001 Missing type annotation for argument 'data'
```

Claude should:
1. Read the file
2. Add appropriate docstring following Google style
3. Add type annotation based on usage context
4. Verify syntax is correct
5. Save the file

## Exclusions

- Don't add type hints to third-party library overrides unless certain
- Don't add docstrings to obvious property getters/setters
- Don't fix S101 in test files
- Don't add type hints to `*args, **kwargs` unless the types are clear
