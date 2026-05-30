# Add Package Skill

Creates new package in `packages/` directory with full workspace integration, structure, tests, and documentation.

## Usage

```
/add-package
```

## What This Skill Does

1. Prompts for package details (name, type, dependencies)
2. Creates complete package directory structure
3. Generates pyproject.toml with uv_build backend
4. Creates package __init__.py with exports
5. Creates README with documentation sections
6. Creates basic module and test templates
7. Updates root pyproject.toml workspace members
8. Runs automatic validation (sync, import, version, tests)
9. Stages files with git (user handles commit)

## Execution Steps

### Step 1: Gather Information

Use the `AskUserQuestion` tool to collect package details:

**Questions to ask:**

1. **Package Name** (header: "Name")
   - Question: "What is the package name? (lowercase with underscores, e.g., 'rltools', 'evalkit', 'dataprep')"
   - Options:
     - label: "rltools", description: "Reinforcement learning tools (RLHF, PPO, DPO)"
     - label: "evalkit", description: "Evaluation utilities and benchmarks"
     - label: "dataprep", description: "Data preparation and preprocessing"
     - label: "agents", description: "Agent frameworks and tools"

2. **Description** (header: "Description")
   - Question: "One-sentence description of what this package does?"
   - Options:
     - label: "Reinforcement learning tools for LLM alignment", description: "RL/RLHF package"
     - label: "Evaluation benchmarks and metrics", description: "Eval package"
     - label: "Data preprocessing and tokenization", description: "Data package"
     - label: "Agent orchestration and tools", description: "Agents package"

3. **Package Type** (header: "Type")
   - Question: "What type of package is this?"
   - Options:
     - label: "Library", description: "Reusable library code (most common)"
     - label: "Tools", description: "CLI tools and utilities"
     - label: "Analysis", description: "Analysis and visualization tools"

4. **Dependencies** (header: "Dependencies", multiSelect: true)
   - Question: "What external dependencies does this package need?"
   - Options:
     - label: "torch", description: "PyTorch for neural networks"
     - label: "numpy", description: "NumPy for numerical operations"
     - label: "pandas", description: "Pandas for data manipulation"
     - label: "datasets", description: "HuggingFace datasets"
   - multiSelect: true

5. **Workspace Dependencies** (header: "Workspace", multiSelect: true)
   - Question: "Which workspace packages does this depend on?"
   - Options:
     - label: "llmkit", description: "Core LLM toolkit"
     - label: "exptools", description: "Experiment analysis tools"
   - multiSelect: true

6. **Entry Points** (header: "Scripts")
   - Question: "Does this package provide CLI scripts?"
   - Options:
     - label: "No scripts", description: "Library only, no CLI tools"
     - label: "Single script", description: "One main CLI tool"
     - label: "Multiple scripts", description: "Several CLI tools"

### Step 2: Validate Package Name

Validate the package name before proceeding:

```bash
PACKAGE_NAME="{PACKAGE_NAME}"  # e.g., "rltools"

# Check if package already exists
if [ -d "packages/${PACKAGE_NAME}" ]; then
    echo "⚠️  Package ${PACKAGE_NAME} already exists"
    # Ask user if they want to overwrite
fi

# Validate naming (lowercase, underscores only)
if ! echo "${PACKAGE_NAME}" | grep -qE '^[a-z][a-z0-9_]*$'; then
    echo "❌ Package name must be lowercase with underscores (e.g., 'rltools', 'eval_kit')"
    exit 1
fi

echo "✓ Package name valid: ${PACKAGE_NAME}"
```

### Step 3: Create Directory Structure

Create the complete package structure:

```bash
PACKAGE_NAME="{PACKAGE_NAME}"
PACKAGE_DIR="packages/${PACKAGE_NAME}"

# Create package structure
mkdir -p "${PACKAGE_DIR}/${PACKAGE_NAME}"
mkdir -p "${PACKAGE_DIR}/tests/unit"
mkdir -p "${PACKAGE_DIR}/tests/integration"
mkdir -p "${PACKAGE_DIR}/docs"

echo "✓ Created directory structure"
tree "${PACKAGE_DIR}"
```

### Step 4: Create pyproject.toml

Use the `Write` tool to create the package configuration.

**Template for `pyproject.toml`:**

```toml
[build-system]
requires = ["uv_build>=0.11.17,<0.12"]
build-backend = "uv_build"

[project]
name = "{PACKAGE_NAME}"
version = "0.1.0"
description = "{DESCRIPTION}"
readme = "README.md"
requires-python = ">=3.11"
authors = [
    {name = "{AUTHOR_NAME}"},
]
dependencies = [
{DEPENDENCIES}
]

{WORKSPACE_DEPENDENCIES}

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
]

{SCRIPTS_SECTION}

[tool.uv.build-backend]
module-root = "."

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

[tool.ruff]
line-length = 120
target-version = "py311"
```

**Dependencies formatting:**
```toml
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]
```

**Workspace dependencies formatting:**
```toml
[tool.uv.sources]
llmkit = { workspace = true }
exptools = { workspace = true }
```

**Scripts section (if applicable):**
```toml
[project.scripts]
{PACKAGE_NAME} = "{PACKAGE_NAME}.cli:main"
{PACKAGE_NAME}-tool = "{PACKAGE_NAME}.cli:tool_main"
```

### Step 5: Create Package __init__.py

**Template for `{package_name}/__init__.py`:**

```python
"""{DESCRIPTION}

{DETAILED_DESCRIPTION}

Usage:
    from {PACKAGE_NAME} import {MAIN_CLASS}

    # Example usage
    {USAGE_EXAMPLE}
"""

__version__ = "0.1.0"

{IMPORTS}

__all__ = [
    "__version__",
{EXPORTS}
]
```

**Example for rltools:**
```python
"""Reinforcement learning tools for LLM alignment.

This package provides tools for training language models with reinforcement
learning, including RLHF, PPO, and DPO implementations.

Usage:
    from rltools import RLTrainer, PPOConfig

    config = PPOConfig(
        model_name="gpt2",
        learning_rate=1e-5,
    )

    trainer = RLTrainer(config)
    trainer.train()
"""

__version__ = "0.1.0"

from .config import PPOConfig, DPOConfig, RLConfig
from .trainer import RLTrainer
from .rewards import RewardModel
from .ppo import PPOTrainer
from .dpo import DPOTrainer

__all__ = [
    "__version__",
    # Config
    "PPOConfig",
    "DPOConfig",
    "RLConfig",
    # Trainers
    "RLTrainer",
    "PPOTrainer",
    "DPOTrainer",
    # Rewards
    "RewardModel",
]
```

### Step 6: Create Basic Module Template

**Template for `{package_name}/core.py` (or main module):**

```python
"""{PACKAGE_NAME} core functionality.

{MODULE_DESCRIPTION}
"""
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class {MAIN_CLASS}:
    """{MAIN_CLASS_DESCRIPTION}

    Args:
        config: Configuration object
        **kwargs: Additional keyword arguments
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize {MAIN_CLASS}.

        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration
        """
        self.config = config or {}
        self.config.update(kwargs)

    def run(self):
        """Main entry point for {MAIN_CLASS}.

        Returns:
            Result of operation
        """
        raise NotImplementedError("Subclasses must implement run()")


def {HELPER_FUNCTION}():
    """{HELPER_DESCRIPTION}

    Returns:
        {RETURN_DESCRIPTION}
    """
    pass
```

### Step 7: Create README.md

**Template for `README.md`:**

```markdown
# {PACKAGE_NAME}

{DESCRIPTION}

## Installation

This package is part of the frontier-ready workspace.

```bash
# From workspace root
uv sync
```

## Usage

```python
from {PACKAGE_NAME} import {MAIN_CLASS}

# Create instance
{USAGE_EXAMPLE}
```

## API Reference

### {MAIN_CLASS}

{MAIN_CLASS_DESCRIPTION}

**Parameters:**
- `param1` (type): Description
- `param2` (type): Description

**Methods:**
- `method1()`: Description
- `method2()`: Description

### {HELPER_FUNCTION}

{HELPER_DESCRIPTION}

**Parameters:**
- `param1` (type): Description

**Returns:**
- `return_type`: Description

## Examples

### Example 1: {EXAMPLE_TITLE}

```python
{EXAMPLE_CODE}
```

### Example 2: {EXAMPLE_TITLE_2}

```python
{EXAMPLE_CODE_2}
```

## Development

### Running Tests

```bash
cd packages/{PACKAGE_NAME}
uv run pytest tests/ -v
```

### Running with Coverage

```bash
uv run pytest tests/ --cov={PACKAGE_NAME} --cov-report=term
```

### Linting

```bash
uv run ruff check {PACKAGE_NAME}/
```

## Attribution

{ATTRIBUTION}
```

### Step 8: Create Test Templates

**Template for `tests/conftest.py`:**

```python
"""Pytest fixtures for {PACKAGE_NAME} tests."""
import pytest
import torch
from {PACKAGE_NAME} import {MAIN_CLASS}


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "param1": "value1",
        "param2": 42,
    }


@pytest.fixture
def {PACKAGE_NAME}_instance(sample_config):
    """Instance of {MAIN_CLASS} for testing."""
    return {MAIN_CLASS}(sample_config)


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")
```

**Template for `tests/unit/test_core.py`:**

```python
"""Tests for {PACKAGE_NAME} core functionality."""
import pytest
import torch
from {PACKAGE_NAME}.core import {MAIN_CLASS}, {HELPER_FUNCTION}


class Test{MAIN_CLASS}:
    """Tests for {MAIN_CLASS}."""

    def test_initialization(self, sample_config):
        """Test {MAIN_CLASS} initializes correctly."""
        instance = {MAIN_CLASS}(sample_config)
        assert instance is not None
        assert instance.config == sample_config

    def test_initialization_with_kwargs(self):
        """Test initialization with keyword arguments."""
        instance = {MAIN_CLASS}(param1="value", param2=100)
        assert instance.config["param1"] == "value"
        assert instance.config["param2"] == 100

    def test_run_method_exists(self, {PACKAGE_NAME}_instance):
        """Test run method exists."""
        assert hasattr({PACKAGE_NAME}_instance, "run")


class Test{HELPER_FUNCTION_CAMEL}:
    """Tests for {HELPER_FUNCTION} helper function."""

    def test_{HELPER_FUNCTION}_exists(self):
        """Test {HELPER_FUNCTION} function exists."""
        assert callable({HELPER_FUNCTION})
```

**Template for `tests/integration/test_workflow.py`:**

```python
"""Integration tests for {PACKAGE_NAME} workflows."""
import pytest
from {PACKAGE_NAME} import {MAIN_CLASS}


class TestWorkflow:
    """Tests for end-to-end workflows."""

    def test_basic_workflow(self, sample_config):
        """Test basic workflow executes successfully."""
        instance = {MAIN_CLASS}(sample_config)
        # Add workflow test
        pass
```

### Step 9: Update Workspace Configuration

Update the root `pyproject.toml` to include the new package:

```bash
# Read current workspace members
grep -A 20 "\[tool.uv.workspace\]" pyproject.toml
```

**Use the `Edit` tool to add new workspace member:**

```toml
[tool.uv.workspace]
members = [
    "packages/llmkit",
    "packages/exptools",
    "packages/{PACKAGE_NAME}",  # Add this line
    "experiments/*",
]
```

### Step 10: Automatic Validation

Run validation checks:

```bash
cd /Users/jex/frontier-ready

# 1. Workspace sync
echo "Syncing workspace..."
uv sync

if [ $? -ne 0 ]; then
    echo "❌ Workspace sync failed"
    echo "Check pyproject.toml for errors"
    exit 1
fi
echo "✓ Workspace synced"

# 2. Import validation
echo "Validating imports..."
uv run python -c "
import {PACKAGE_NAME}
print(f'✓ Package imported: {PACKAGE_NAME} v{{PACKAGE_NAME}.__version__}')
"

# 3. Version check
echo "Checking version..."
uv run python -c "
from {PACKAGE_NAME} import __version__
assert __version__ == '0.1.0'
print(f'✓ Version: {{__version__}}')
"

# 4. Test discovery
echo "Discovering tests..."
cd packages/{PACKAGE_NAME}
uv run pytest --collect-only tests/
TEST_COUNT=$(uv run pytest --collect-only tests/ -q | grep "test session starts" -A 100 | grep "test_" | wc -l)
echo "✓ Discovered ${TEST_COUNT} tests"

# 5. Test execution
echo "Running tests..."
uv run pytest tests/ -v

# 6. Linting
echo "Running linter..."
uv run ruff check {PACKAGE_NAME}/

# 7. CLI scripts (if applicable)
if [ -n "{SCRIPTS}" ]; then
    echo "Validating CLI scripts..."
    uv run {PACKAGE_NAME} --help
    echo "✓ CLI script works"
fi

echo ""
echo "✅ All validations passed!"
```

### Step 11: Stage Files with Git

```bash
cd /Users/jex/frontier-ready

# Stage package directory
git add packages/{PACKAGE_NAME}

# Stage workspace config
git add pyproject.toml

# Show what's staged
git status
```

### Step 12: Summary Output

```
✅ Package {PACKAGE_NAME} created successfully!

📁 Package structure:
  packages/{PACKAGE_NAME}/
  ├── {PACKAGE_NAME}/
  │   ├── __init__.py
  │   └── core.py
  ├── tests/
  │   ├── conftest.py
  │   ├── unit/
  │   │   └── test_core.py
  │   └── integration/
  │       └── test_workflow.py
  ├── pyproject.toml
  └── README.md

✓ Validations passed:
  ✓ Workspace sync successful
  ✓ Package imports correctly
  ✓ Version check: 0.1.0
  ✓ Test discovery: {num_tests} tests
  ✓ Test execution: all passed
  ✓ Linting: no issues
  {CLI_VALIDATION}

📝 Suggested commit message:

Add package: {PACKAGE_NAME}

- {DESCRIPTION}
- Type: {PACKAGE_TYPE}
- Dependencies: {DEPENDENCIES_LIST}
- Workspace integration: uv workspace member
- Tests: {num_tests} test cases
- Documentation: README with usage examples

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

📦 Next steps:
  1. Review package: packages/{PACKAGE_NAME}/
  2. Implement core functionality: packages/{PACKAGE_NAME}/{PACKAGE_NAME}/core.py
  3. Add tests: packages/{PACKAGE_NAME}/tests/
  4. Commit: git commit -m "Add package: {PACKAGE_NAME}"
  5. Use in experiments: Add {PACKAGE_NAME} to experiment dependencies

💡 Usage in experiments:

Add to experiment's pyproject.toml:
```toml
dependencies = [
    "llmkit",
    "{PACKAGE_NAME}",
]

[tool.uv.sources]
{PACKAGE_NAME} = {{ workspace = true }}
```

Then use in experiment code:
```python
from {PACKAGE_NAME} import {MAIN_CLASS}

# Your code here
```

📊 Package info:
  Name: {PACKAGE_NAME}
  Version: 0.1.0
  Type: {PACKAGE_TYPE}
  Dependencies: {len(dependencies)} external, {len(workspace_deps)} workspace
  {SCRIPTS_INFO}
```

## Error Handling

### Package Already Exists
```bash
if [ -d "packages/${PACKAGE_NAME}" ]; then
    echo "⚠️  Package ${PACKAGE_NAME} already exists"
    # Use AskUserQuestion
    # Options: "Overwrite", "Choose different name", "Cancel"
fi
```

### Invalid Package Name
```bash
if ! echo "${PACKAGE_NAME}" | grep -qE '^[a-z][a-z0-9_]*$'; then
    echo "❌ Package name must be lowercase with underscores"
    echo "Valid: rltools, eval_kit, data_prep"
    echo "Invalid: RLTools, RL-tools, rl.tools"
    exit 1
fi
```

### Workspace Sync Errors
```bash
if ! uv sync; then
    echo "❌ Workspace sync failed"
    echo "Common issues:"
    echo "  - Dependency conflict"
    echo "  - Invalid pyproject.toml syntax"
    echo "  - Missing workspace member declaration"
    echo ""
    echo "Check error above and fix manually"
    exit 1
fi
```

### Import Errors
```bash
if ! uv run python -c "import ${PACKAGE_NAME}"; then
    echo "❌ Cannot import package"
    echo "Possible issues:"
    echo "  - Workspace not synced (run: uv sync)"
    echo "  - Syntax error in __init__.py"
    echo "  - Missing dependencies"
    exit 1
fi
```

### Test Failures
```bash
if ! uv run pytest tests/ -v; then
    echo "⚠️  Tests failed"
    echo "Review test output and fix failing tests"
    echo "Package is still staged - you can commit and fix later"
fi
```

## Notes

- **uv_build backend:** Use uv_build for package building
- **Workspace members:** Add to root pyproject.toml [tool.uv.workspace] members
- **Workspace dependencies:** Use `{ workspace = true }` in [tool.uv.sources]
- **Version:** Start at 0.1.0, follow semantic versioning
- **Python version:** Use >=3.11 for consistency
- **__all__ exports:** Define public API in __init__.py
- **README:** Include installation, usage, API reference, examples
- **Tests:** Create both unit and integration test directories
- **conftest.py:** Define fixtures for reusable test data
- **CLI scripts:** Add [project.scripts] for command-line tools
- **Linting:** Use ruff with line-length=120
- **Attribution:** Credit sources/inspirations in README

## Reference Files

- **Package structure:** `packages/llmkit/` and `packages/exptools/`
- **pyproject.toml:** `packages/llmkit/pyproject.toml`
- **Package init:** `packages/llmkit/llmkit/__init__.py`
- **Tests:** `packages/llmkit/tests/`
- **Workspace config:** `pyproject.toml` (root)
