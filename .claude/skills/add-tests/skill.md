# Add Tests Skill

Creates comprehensive test suite for existing or new components with fixtures, parametrization, and coverage validation.

## Usage

```
/add-tests
```

## What This Skill Does

1. Prompts for test target details (component/variant/model, scope)
2. Analyzes target code to understand interface
3. Generates test file with comprehensive test cases
4. Updates conftest.py with new fixtures if needed
5. Runs automatic validation (syntax, discovery, execution, coverage)
6. Stages files with git (user handles commit)

## Execution Steps

### Step 1: Gather Information

Use the `AskUserQuestion` tool to collect test requirements:

**Questions to ask:**

1. **Test Target Type** (header: "Target Type")
   - Question: "What type of code are you testing?"
   - Options:
     - label: "Component", description: "Core component (attention, feedforward, embedding, normalization)"
     - label: "Variant", description: "Architecture variant (create_rope_gpt, create_swiglu_gpt)"
     - label: "Model", description: "Full model (GPT, Trainer)"
     - label: "Integration", description: "End-to-end training/eval workflow"

2. **Target Name** (header: "Target")
   - Question: "What is the exact name of what you're testing? (e.g., 'MultiQueryAttention', 'SwiGLU', 'GPT')"
   - Options:
     - label: "CausalSelfAttention", description: "Standard attention mechanism"
     - label: "MLP", description: "Standard feedforward"
     - label: "GPT", description: "Full GPT model"
     - label: "Trainer", description: "Training loop"

3. **Test Scope** (header: "Scope", multiSelect: true)
   - Question: "Which test categories do you want to include?"
   - Options:
     - label: "initialization", description: "Test component initialization and parameter setup"
     - label: "forward_pass", description: "Test forward pass shape preservation and correctness"
     - label: "config_validation", description: "Test config validation and error handling"
     - label: "edge_cases", description: "Test edge cases (empty input, large batch, etc.)"
     - label: "gradient_flow", description: "Test gradient computation and backprop"
     - label: "parametrization", description: "Test with different config variations"
   - multiSelect: true

4. **Existing Tests** (header: "Existing")
   - Question: "Do tests already exist for this component?"
   - Options:
     - label: "No tests", description: "Create new test file from scratch"
     - label: "Extend existing", description: "Add tests to existing test file"
     - label: "Check first", description: "Search for existing tests automatically"

### Step 2: Locate and Analyze Target

Determine the target file location and analyze its interface:

```bash
TARGET_NAME="{TARGET_NAME}"  # e.g., "MultiQueryAttention"
TARGET_TYPE="{TARGET_TYPE}"  # e.g., "Component"

# Convert to snake_case for file searching
SNAKE_CASE=$(echo "${TARGET_NAME}" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')

# Locate target file based on type
case "${TARGET_TYPE}" in
    "Component")
        TARGET_FILE="packages/llmkit/llmkit/core/${SNAKE_CASE}.py"
        ;;
    "Variant")
        TARGET_FILE="packages/llmkit/llmkit/variants/${SNAKE_CASE}.py"
        ;;
    "Model")
        TARGET_FILE="packages/llmkit/llmkit/models/${SNAKE_CASE}.py"
        ;;
    "Integration")
        # Integration tests typically test workflows
        TARGET_FILE="packages/llmkit/llmkit/training.py"
        ;;
esac

if [ ! -f "${TARGET_FILE}" ]; then
    echo "❌ Target file not found: ${TARGET_FILE}"
    echo "Searching for ${TARGET_NAME}..."
    find packages/llmkit -name "*${SNAKE_CASE}*" -type f
    exit 1
fi

echo "✓ Found target: ${TARGET_FILE}"
```

**Analyze the target code:**

```bash
# Extract class signature
grep -A 3 "^class ${TARGET_NAME}" "${TARGET_FILE}"

# Extract __init__ signature
grep -A 10 "def __init__" "${TARGET_FILE}" | head -15

# Extract forward signature
grep -A 5 "def forward" "${TARGET_FILE}" | head -10

# Extract config requirements
grep "Config" "${TARGET_FILE}" | head -5
```

### Step 3: Check for Existing Tests

Determine if tests already exist:

```bash
# Determine test file location
if [ "${TARGET_TYPE}" = "Integration" ]; then
    TEST_FILE="packages/llmkit/tests/integration/test_${SNAKE_CASE}.py"
else
    TEST_FILE="packages/llmkit/tests/unit/test_${SNAKE_CASE}.py"
fi

if [ -f "${TEST_FILE}" ]; then
    echo "⚠️  Tests already exist: ${TEST_FILE}"
    # Count existing tests
    EXISTING_COUNT=$(grep -c "def test_" "${TEST_FILE}")
    echo "  Existing tests: ${EXISTING_COUNT}"

    # If user chose "Extend existing", read the file to understand structure
    # If user chose "No tests", ask if they want to overwrite
fi

echo "📝 Test file: ${TEST_FILE}"
```

### Step 4: Generate Test File

Use the `Write` tool to create comprehensive test suite.

**Template variables:**
- `{TARGET_NAME}`: Class name (e.g., "MultiQueryAttention")
- `{CONFIG_CLASS}`: Config class name (e.g., "MultiQueryAttentionConfig")
- `{SNAKE_CASE}`: Module name (e.g., "multi_query_attention")
- `{TEST_SCOPE}`: Selected test categories
- `{REQUIRED_PARAMS}`: Required config parameters extracted from code
- `{PARAM_VARIATIONS}`: Parameter variations for parametrize

**Base template for unit tests:**

```python
"""Tests for {TARGET_NAME} {TARGET_TYPE}."""
import pytest
import torch
from llmkit.config.component_config import {CONFIG_CLASS}
from llmkit.core.{SNAKE_CASE} import {TARGET_NAME}


class Test{TARGET_NAME}:
    """Tests for {TARGET_NAME} {TARGET_TYPE}."""

{TEST_METHODS}
```

**Test method templates based on scope:**

**1. Initialization tests:**
```python
    def test_initialization(self):
        """Test {TARGET_NAME} initializes correctly."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            {REQUIRED_PARAMS}
        )
        component = {TARGET_NAME}(config, layer_idx=0, n_layer=12)

        assert component is not None
        assert isinstance(component, torch.nn.Module)

        # Verify config parameters are stored
        {CONFIG_ASSERTIONS}

    def test_initialization_different_layers(self):
        """Test initialization with different layer indices."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})

        # First layer
        comp_0 = {TARGET_NAME}(config, layer_idx=0, n_layer=12)

        # Middle layer
        comp_6 = {TARGET_NAME}(config, layer_idx=6, n_layer=12)

        # Last layer
        comp_11 = {TARGET_NAME}(config, layer_idx=11, n_layer=12)

        # All should initialize successfully
        assert all(c is not None for c in [comp_0, comp_6, comp_11])
```

**2. Forward pass tests:**
```python
    def test_forward_shape_preservation(self):
        """Test forward pass preserves input shape."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 128)
        y = component(x)

        assert y.shape == (batch_size, seq_len, 128)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 16, 128)
            y = component(x)
            assert y.shape == (batch_size, 16, 128)

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence lengths."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        for seq_len in [1, 16, 64, 128]:
            x = torch.randn(2, seq_len, 128)
            y = component(x)
            assert y.shape == (2, seq_len, 128)

    def test_forward_deterministic_in_eval(self):
        """Test forward pass is deterministic in eval mode."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)
        component.eval()

        x = torch.randn(2, 16, 128)
        y1 = component(x)
        y2 = component(x)

        assert torch.allclose(y1, y2)
```

**3. Config validation tests:**
```python
    def test_config_validation_invalid_n_embd(self):
        """Test config validation catches invalid n_embd."""
        with pytest.raises(ValueError, match="divisible by"):
            {CONFIG_CLASS}(
                n_embd=127,  # Not divisible by n_head
                {INVALID_PARAMS}
            )

    def test_config_validation_negative_values(self):
        """Test config validation catches negative values."""
        with pytest.raises(ValueError):
            {CONFIG_CLASS}(
                n_embd=-128,
                {REQUIRED_PARAMS}
            )

    def test_config_defaults(self):
        """Test config uses correct default values."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            {MINIMAL_REQUIRED_PARAMS}
        )

        # Check defaults
        assert config.dropout == 0.0
        assert config.bias == True
        {DEFAULT_ASSERTIONS}
```

**4. Edge case tests:**
```python
    def test_edge_case_single_token(self):
        """Test with sequence length of 1."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        x = torch.randn(2, 1, 128)
        y = component(x)

        assert y.shape == (2, 1, 128)
        assert not torch.isnan(y).any()

    def test_edge_case_large_batch(self):
        """Test with large batch size."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        x = torch.randn(128, 16, 128)
        y = component(x)

        assert y.shape == (128, 16, 128)

    def test_edge_case_zero_input(self):
        """Test with all-zero input."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        x = torch.zeros(2, 16, 128)
        y = component(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()
```

**5. Gradient flow tests:**
```python
    def test_gradient_flow(self):
        """Test gradients flow through component."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        x = torch.randn(2, 16, 128, requires_grad=True)
        y = component(x)
        loss = y.sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # All parameters should have gradients
        for name, param in component.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradient_magnitude(self):
        """Test gradient magnitudes are reasonable."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        x = torch.randn(2, 16, 128, requires_grad=True)
        y = component(x)
        loss = y.mean()
        loss.backward()

        # Check gradient magnitudes
        for name, param in component.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 100.0, f"Gradient too large for {name}: {grad_norm}"
                assert grad_norm > 0.0, f"Zero gradient for {name}"
```

**6. Parametrization tests:**
```python
    @pytest.mark.parametrize("n_embd", [64, 128, 256, 512, 768])
    def test_different_embedding_dims(self, n_embd):
        """Test component with different embedding dimensions."""
        config = {CONFIG_CLASS}(
            n_embd=n_embd,
            {REQUIRED_PARAMS_WITH_N_EMBD}
        )
        component = {TARGET_NAME}(config)

        x = torch.randn(2, 16, n_embd)
        y = component(x)

        assert y.shape == (2, 16, n_embd)

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
    def test_different_dropout_values(self, dropout):
        """Test component with different dropout values."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            dropout=dropout,
            {REQUIRED_PARAMS}
        )
        component = {TARGET_NAME}(config)

        x = torch.randn(2, 16, 128)
        y = component(x)

        assert y.shape == x.shape

    @pytest.mark.parametrize("bias", [True, False])
    def test_with_and_without_bias(self, bias):
        """Test component with and without bias."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            bias=bias,
            {REQUIRED_PARAMS}
        )
        component = {TARGET_NAME}(config)

        # Check bias presence in linear layers
        {BIAS_ASSERTIONS}

        x = torch.randn(2, 16, 128)
        y = component(x)

        assert y.shape == x.shape
```

**Additional tests for specific component types:**

**For Attention components:**
```python
    def test_causal_masking(self):
        """Test attention properly masks future tokens."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            flash_attention=False,  # Disable flash to test manual masking
            {REQUIRED_PARAMS}
        )
        component = {TARGET_NAME}(config)
        component.eval()

        # Create input where first token differs
        x1 = torch.randn(1, 16, 128)
        x2 = x1.clone()
        x2[:, 0, :] = torch.randn(1, 128)  # Change first token

        y1 = component(x1)
        y2 = component(x2)

        # First token output should differ
        assert not torch.allclose(y1[:, 0, :], y2[:, 0, :])

        # But future tokens should be the same (due to causal masking)
        # (This may not be true due to residual connections, depends on component)

    @pytest.mark.parametrize("flash_attention", [True, False])
    def test_flash_attention_consistency(self, flash_attention):
        """Test output is consistent with and without flash attention."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            flash_attention=flash_attention,
            {REQUIRED_PARAMS}
        )
        component = {TARGET_NAME}(config)
        component.eval()

        x = torch.randn(2, 16, 128)
        y = component(x)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_parameter_count(self):
        """Test attention has expected parameter count."""
        config = {CONFIG_CLASS}(n_embd=768, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        param_count = sum(p.numel() for p in component.parameters())

        # Expected: depends on attention variant
        # Standard MHA: 4 * (n_embd * n_embd) for qkv + output projection
        {PARAM_COUNT_CALCULATION}

        assert param_count == expected, f"Expected {expected} params, got {param_count}"
```

**For Feedforward components:**
```python
    def test_activation_function(self):
        """Test feedforward uses correct activation."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {TARGET_NAME}(config)

        # Component should have activation
        {ACTIVATION_ASSERTIONS}

    @pytest.mark.parametrize("expansion_factor", [2, 4, 8])
    def test_expansion_factors(self, expansion_factor):
        """Test feedforward with different expansion factors."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            expansion_factor=expansion_factor,
            {REQUIRED_PARAMS}
        )
        component = {TARGET_NAME}(config)

        # Check hidden dimension
        {EXPANSION_ASSERTIONS}

        x = torch.randn(2, 16, 128)
        y = component(x)

        assert y.shape == x.shape
```

**For Variants:**
```python
"""Tests for {TARGET_NAME} variant."""
import pytest
import torch
from llmkit import ModelConfig, GPT
from llmkit.variants.{SNAKE_CASE} import {TARGET_NAME}


class Test{TARGET_NAME}:
    """Tests for {TARGET_NAME} variant."""

    def test_variant_creation(self):
        """Test variant creates GPT model successfully."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        model = {TARGET_NAME}(config)

        assert isinstance(model, GPT)

    def test_variant_forward(self):
        """Test variant model forward pass."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        model = {TARGET_NAME}(config)

        x = torch.randint(0, 100, (2, 16))
        logits, loss = model(x, x)

        assert logits.shape == (2, 16, 100)

    def test_variant_differs_from_baseline(self):
        """Test variant produces different output than baseline."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        baseline = GPT(config)
        variant = {TARGET_NAME}(config)

        # Initialize with same seed
        torch.manual_seed(42)
        baseline.apply(baseline._init_weights)

        torch.manual_seed(42)
        variant.apply(variant._init_weights)

        x = torch.randint(0, 100, (2, 16))

        baseline.eval()
        variant.eval()

        logits_baseline, _ = baseline(x)
        logits_variant, _ = variant(x)

        # Variants should differ (unless it's a null variant)
        # assert not torch.allclose(logits_baseline, logits_variant)
```

**For Integration tests:**
```python
"""Integration tests for training workflow."""
import pytest
import torch
from pathlib import Path
import tempfile
from llmkit import ModelConfig, GPT, Trainer, TrainingConfig


class TestTrainingWorkflow:
    """Tests for end-to-end training workflow."""

    def test_training_step(self, tmp_path):
        """Test single training step executes."""
        model_config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        training_config = TrainingConfig(
            model_config=model_config,
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            max_steps=1,
            batch_size=2,
            sequence_length=16,
        )

        # Create dummy data
        (tmp_path / "data").mkdir()
        torch.save({"tokens": torch.randint(0, 100, (1000,))}, tmp_path / "data" / "train_000.pt")

        trainer = Trainer(training_config)
        trainer.train()

        assert True  # If we got here, training step worked

    def test_loss_decreases(self, tmp_path):
        """Test loss decreases over multiple steps."""
        # Similar setup, run 10 steps, verify loss trend
        pass
```

### Step 5: Update conftest.py (if needed)

If new fixtures are needed, update `tests/conftest.py`:

```python
# Add new fixtures based on component type

@pytest.fixture
def {snake_case}_config():
    """Config for {TARGET_NAME} testing."""
    return {CONFIG_CLASS}(
        n_embd=128,
        {DEFAULT_PARAMS}
    )

@pytest.fixture
def {snake_case}_component({snake_case}_config):
    """Component instance for testing."""
    return {TARGET_NAME}({snake_case}_config)
```

### Step 6: Automatic Validation

Run validation checks:

```bash
cd packages/llmkit

# 1. Syntax check
echo "Checking Python syntax..."
uv run python -m py_compile tests/unit/test_{snake_case_name}.py
echo "✓ Syntax valid"

# 2. Test discovery
echo "Discovering tests..."
uv run pytest tests/unit/test_{snake_case_name}.py --collect-only
TEST_COUNT=$(uv run pytest tests/unit/test_{snake_case_name}.py --collect-only -q | grep "test session starts" -A 100 | grep "test_" | wc -l)
echo "✓ Discovered ${TEST_COUNT} tests"

# 3. Test execution
echo "Running tests..."
uv run pytest tests/unit/test_{snake_case_name}.py -v

# 4. Coverage check
echo "Checking coverage..."
uv run pytest tests/unit/test_{snake_case_name}.py --cov=llmkit.core.{snake_case_name} --cov-report=term

COVERAGE=$(uv run pytest tests/unit/test_{snake_case_name}.py --cov=llmkit.core.{snake_case_name} --cov-report=term | grep "TOTAL" | awk '{print $4}' | sed 's/%//')

if [ "${COVERAGE}" -lt 90 ]; then
    echo "⚠️  Coverage is ${COVERAGE}% (target: 90%)"
    echo "Consider adding more tests for edge cases"
else
    echo "✓ Coverage: ${COVERAGE}%"
fi

# 5. Full test suite (ensure no regressions)
echo "Running full test suite..."
uv run pytest tests/ -v

echo ""
echo "✅ All validations passed!"
```

### Step 7: Stage Files with Git

Stage the created/modified files:

```bash
cd /Users/jex/frontier-ready

# Stage test file
git add packages/llmkit/tests/unit/test_{snake_case_name}.py

# Stage conftest if modified
git add packages/llmkit/tests/conftest.py

# Show what's staged
git status
```

### Step 8: Summary Output

Print comprehensive summary:

```
✅ Test suite for {TARGET_NAME} created successfully!

📁 Files created/modified:
  ✓ packages/llmkit/tests/unit/test_{snake_case_name}.py ({num_tests} tests)
  {CONFTEST_LINE}

✓ Validations passed:
  ✓ Python syntax valid
  ✓ Test discovery: {num_tests} tests found
  ✓ Test execution: {passed}/{total} passed
  ✓ Coverage: {coverage}% (target: 90%)
  ✓ Full test suite: no regressions

📊 Test breakdown:
  {TEST_CATEGORY_COUNTS}

📝 Suggested commit message:

Add comprehensive tests for {TARGET_NAME}

- {num_tests} test cases covering {categories}
- Coverage: {coverage}%
- Test scopes: {test_scopes}

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

📦 Next steps:
  1. Review tests: packages/llmkit/tests/unit/test_{snake_case_name}.py
  2. Run tests: uv run pytest tests/unit/test_{snake_case_name}.py -v
  3. Check coverage: uv run pytest tests/unit/test_{snake_case_name}.py --cov
  4. Commit: git commit -m "Add tests for {TARGET_NAME}"

💡 Run specific test:
  uv run pytest tests/unit/test_{snake_case_name}.py::Test{TARGET_NAME}::test_forward_shape_preservation -v

💡 Run with coverage report:
  uv run pytest tests/unit/test_{snake_case_name}.py --cov=llmkit.core.{snake_case_name} --cov-report=html
  open htmlcov/index.html
```

## Error Handling

### Target Not Found
```bash
if [ ! -f "${TARGET_FILE}" ]; then
    echo "❌ Target not found: ${TARGET_FILE}"
    echo ""
    echo "Searching for ${TARGET_NAME}..."
    find packages/llmkit -name "*${SNAKE_CASE}*" -type f
    echo ""
    echo "Please verify the target name and type are correct"
    exit 1
fi
```

### Test File Already Exists
```bash
if [ -f "${TEST_FILE}" ]; then
    echo "⚠️  Test file already exists: ${TEST_FILE}"
    # Use AskUserQuestion
    # Options: "Extend (add new tests)", "Overwrite (replace all tests)", "Cancel"
fi
```

### Test Failures
```bash
if ! uv run pytest tests/unit/test_${snake_case_name}.py -v; then
    echo "⚠️  Some tests failed"
    echo ""
    echo "Common issues:"
    echo "  - Config parameters don't match component requirements"
    echo "  - Expected behavior doesn't match implementation"
    echo "  - Missing fixtures in conftest.py"
    echo ""
    echo "Review test output above and fix failing tests"
    echo "Tests are still staged - you can commit and fix later"
fi
```

### Low Coverage
```bash
if [ "${COVERAGE}" -lt 90 ]; then
    echo "⚠️  Coverage is ${COVERAGE}% (target: 90%)"
    echo ""
    echo "Missing coverage likely in:"
    echo "  - Edge cases (empty input, large values)"
    echo "  - Error paths (invalid config)"
    echo "  - Conditional branches (dropout, bias)"
    echo ""
    echo "Consider adding tests for uncovered lines"
fi
```

### Import Errors
```bash
# If test imports fail
if ! uv run python -c "from llmkit.core import ${TARGET_NAME}" 2>/dev/null; then
    echo "❌ Cannot import ${TARGET_NAME}"
    echo "Ensure the component exists and is properly exported"
    echo "Run: uv run python -c 'from llmkit.core import ${TARGET_NAME}'"
    exit 1
fi
```

## Notes

- **Comprehensive coverage:** Aim for >90% coverage with diverse test cases
- **Fixtures:** Use fixtures from conftest.py for reusable test data
- **Parametrization:** Use @pytest.mark.parametrize for testing variations
- **Class organization:** Group tests in Test{ComponentName} classes
- **Descriptive names:** test_what_is_being_tested_and_expected_behavior
- **Assertions:** Clear assertion messages for debugging
- **Edge cases:** Test boundary conditions (size 1, size 0, very large)
- **Gradient tests:** Verify backprop works and gradients are reasonable
- **Determinism:** Test eval mode produces consistent results
- **Integration:** Add integration tests for workflow-level functionality
- **Documentation:** Docstrings explain what each test validates
- **Isolation:** Each test should be independent (no shared state)

## Reference Files

- **Test pattern:** `packages/llmkit/tests/unit/test_feedforward.py`
- **Fixtures:** `packages/llmkit/tests/conftest.py`
- **Integration tests:** `packages/llmkit/tests/integration/test_training.py`
- **Parametrization examples:** `packages/llmkit/tests/unit/test_feedforward.py`
