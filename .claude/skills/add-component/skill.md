# Add Component Skill

Adds a new component to `llmkit/core/` with proper abstractions, configuration integration, and comprehensive tests.

## Usage

```
/add-component
```

## What This Skill Does

1. Prompts for component details (type, name, config parameters)
2. Creates component module in `llmkit/core/`
3. Updates or creates config dataclass in `component_config.py`
4. Updates package exports (`__init__.py`)
5. Generates comprehensive test suite
6. Runs automatic validation (imports, tests, linting)
7. Stages files with git (user handles commit)

## Execution Steps

### Step 1: Gather Information

Use the `AskUserQuestion` tool to collect component details:

**Questions to ask:**

1. **Component Type** (header: "Type")
   - Question: "What type of component are you adding?"
   - Options:
     - label: "Attention", description: "Attention mechanism (e.g., Multi-Query, Grouped-Query, Sliding Window)"
     - label: "Feedforward", description: "Feedforward/MLP variant (e.g., SwiGLU, GLU, MoE)"
     - label: "Position Embedding", description: "Position embedding scheme (e.g., RoPE, ALiBi, Learned)"
     - label: "Normalization", description: "Normalization layer (e.g., RMSNorm, LayerScale)"

2. **Component Name** (header: "Name")
   - Question: "What is the component class name? (CamelCase, e.g., 'MultiQueryAttention', 'SwiGLU', 'RoPEEmbedding')"
   - Options:
     - label: "MultiQueryAttention", description: "Multi-query attention with shared KV heads"
     - label: "GroupedQueryAttention", description: "Grouped-query attention (GQA)"
     - label: "SwiGLU", description: "SwiGLU activation MLP"
     - label: "RoPEEmbedding", description: "Rotary position embeddings"

3. **Description** (header: "Description")
   - Question: "One-sentence description of what this component does?"
   - Options:
     - label: "Reduces KV cache size", description: "Memory-efficient attention variant"
     - label: "Improves activation", description: "Better activation function for MLP"
     - label: "Better position encoding", description: "Improved position information"
     - label: "Faster normalization", description: "More efficient normalization"

4. **Base Class** (header: "Base")
   - Question: "What should this component inherit from?"
   - Options:
     - label: "nn.Module", description: "Standard PyTorch module (most common)"
     - label: "PositionEmbedding", description: "Abstract base class for position embeddings"
     - label: "Replace existing", description: "This replaces an existing component (e.g., swap CausalSelfAttention)"

5. **Config Parameters** (header: "Config")
   - Question: "What config parameters does this component need? (comma-separated with types, e.g., 'num_kv_heads: int, window_size: int')"
   - Options:
     - label: "num_kv_heads: int", description: "For multi-query or grouped-query attention"
     - label: "rope_theta: float", description: "For RoPE base frequency"
     - label: "window_size: int", description: "For sliding window attention"
     - label: "eps: float", description: "For normalization epsilon"

### Step 2: Validate Component Name

Before proceeding, validate the component name:

```bash
# Check if component already exists
SNAKE_CASE_NAME=$(echo "{COMPONENT_NAME}" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')

if [ -f "packages/llmkit/llmkit/core/${SNAKE_CASE_NAME}.py" ]; then
    echo "⚠️  Component ${COMPONENT_NAME} already exists at core/${SNAKE_CASE_NAME}.py"
    # Ask user if they want to overwrite
fi

# Validate naming (CamelCase)
if ! echo "{COMPONENT_NAME}" | grep -qE '^[A-Z][a-zA-Z0-9]*$'; then
    echo "❌ Component name must be CamelCase (e.g., 'MultiQueryAttention', 'SwiGLU')"
    exit 1
fi

echo "✓ Component name valid: {COMPONENT_NAME} (core/${SNAKE_CASE_NAME}.py)"
```

### Step 3: Determine File Locations

Based on the component name, determine all file paths:

```bash
COMPONENT_NAME="{COMPONENT_NAME}"  # e.g., "MultiQueryAttention"
SNAKE_CASE_NAME=$(echo "${COMPONENT_NAME}" | sed 's/\([A-Z]\)/_\1/g' | sed 's/^_//' | tr '[:upper:]' '[:lower:]')  # e.g., "multi_query_attention"
CONFIG_CLASS="${COMPONENT_NAME}Config"  # e.g., "MultiQueryAttentionConfig"

# File paths
COMPONENT_FILE="packages/llmkit/llmkit/core/${SNAKE_CASE_NAME}.py"
CONFIG_FILE="packages/llmkit/llmkit/config/component_config.py"
CORE_INIT="packages/llmkit/llmkit/core/__init__.py"
TEST_FILE="packages/llmkit/tests/unit/test_${SNAKE_CASE_NAME}.py"

echo "📁 Files to create/modify:"
echo "  - Component: ${COMPONENT_FILE}"
echo "  - Config: ${CONFIG_FILE} (update)"
echo "  - Exports: ${CORE_INIT} (update)"
echo "  - Tests: ${TEST_FILE}"
```

### Step 4: Create Component Implementation

Use the `Write` tool to create the component module.

**Template variables:**
- `{COMPONENT_NAME}`: CamelCase class name (e.g., "MultiQueryAttention")
- `{CONFIG_CLASS}`: Config class name (e.g., "MultiQueryAttentionConfig")
- `{DESCRIPTION}`: One-sentence description
- `{BASE_CLASS}`: Inheritance (e.g., "nn.Module", "PositionEmbedding")
- `{CONFIG_PARAMS}`: Config parameter assignments from constructor

**Template for `core/{snake_case_name}.py`:**

```python
"""{DESCRIPTION}"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..config.component_config import {CONFIG_CLASS}
from .initialization import ScaledLinear


class {COMPONENT_NAME}({BASE_CLASS}):
    """{DESCRIPTION}

    This component implements {detailed_explanation}.
    """

    def __init__(self, config: {CONFIG_CLASS}, layer_idx: int = 0, n_layer: int = 1):
        """Initialize {COMPONENT_NAME}.

        Args:
            config: {CONFIG_CLASS} with component parameters
            layer_idx: Index of this layer (for residual scaling)
            n_layer: Total number of layers (for residual scaling)
        """
        super().__init__()

        # Store config parameters
        {CONFIG_PARAMS}

        # Initialize component layers
        # TODO: Add component-specific initialization

        # Output projection with residual scaling (if applicable)
        scale_factor = (2 * n_layer) ** -0.5
        # TODO: Add ScaledLinear for output projection if needed
        # self.output_proj = ScaledLinear(
        #     config.n_embd,
        #     config.n_embd,
        #     scale_factor=scale_factor,
        #     bias=config.bias,
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply {COMPONENT_NAME}.

        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        # TODO: Implement forward pass
        return x
```

**Specific templates based on component type:**

**For Attention components:**
```python
"""Multi-query attention with shared KV heads."""
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..config.component_config import MultiQueryAttentionConfig
from .initialization import ScaledLinear


class MultiQueryAttention(nn.Module):
    """Multi-query attention with shared KV heads.

    Reduces KV cache size by sharing key-value projections across all query heads.
    """

    def __init__(self, config: MultiQueryAttentionConfig, layer_idx: int = 0, n_layer: int = 1):
        """Initialize multi-query attention.

        Args:
            config: MultiQueryAttentionConfig with n_embd, n_head, num_kv_heads
            layer_idx: Index of this layer (for residual scaling)
            n_layer: Total number of layers (for residual scaling)
        """
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.num_kv_heads = config.num_kv_heads
        self.dropout = config.dropout
        self.flash_attention = config.flash_attention

        # Query projection (full heads)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Key-value projections (shared across query heads)
        head_dim = config.n_embd // config.n_head
        self.k_proj = nn.Linear(config.n_embd, config.num_kv_heads * head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.num_kv_heads * head_dim, bias=config.bias)

        # Output projection with residual scaling
        scale_factor = (2 * n_layer) ** -0.5
        self.c_proj = ScaledLinear(
            config.n_embd,
            config.n_embd,
            scale_factor=scale_factor,
            bias=config.bias,
        )

        # Dropout
        if self.dropout > 0:
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-query attention.

        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        B, T, C = x.size()
        head_dim = C // self.n_head

        # Project queries, keys, values
        q = self.q_proj(x).view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, head_dim).transpose(1, 2)  # (B, num_kv_heads, T, head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, head_dim).transpose(1, 2)  # (B, num_kv_heads, T, head_dim)

        # Expand KV to match query heads (repeat each KV head)
        n_rep = self.n_head // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=1)  # (B, n_head, T, head_dim)
        v = v.repeat_interleave(n_rep, dim=1)  # (B, n_head, T, head_dim)

        # Causal self-attention
        if self.flash_attention:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim ** 0.5))
            att = att.masked_fill(
                torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1),
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            if self.dropout > 0:
                att = self.attn_dropout(att)
            y = att @ v

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        if self.dropout > 0:
            y = self.resid_dropout(y)

        return y
```

**For Feedforward components (SwiGLU example):**
```python
"""SwiGLU activation feedforward network."""
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..config.component_config import SwiGLUConfig
from .initialization import ScaledLinear


class SwiGLU(nn.Module):
    """Feedforward network with SwiGLU activation.

    Uses gated linear units with SiLU (Swish) activation for improved performance.
    """

    def __init__(self, config: SwiGLUConfig, layer_idx: int = 0, n_layer: int = 1):
        """Initialize SwiGLU feedforward.

        Args:
            config: SwiGLUConfig with n_embd, expansion_factor
            layer_idx: Index of this layer (for residual scaling)
            n_layer: Total number of layers (for residual scaling)
        """
        super().__init__()
        self.n_embd = config.n_embd
        hidden_dim = config.n_embd * config.expansion_factor

        # GLU requires two projections for gating
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)

        # Output projection with residual scaling
        scale_factor = (2 * n_layer) ** -0.5
        self.w_down = ScaledLinear(
            hidden_dim,
            config.n_embd,
            scale_factor=scale_factor,
            bias=config.bias,
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU feedforward.

        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd)
        """
        # SwiGLU: SiLU(W_gate * x) * (W_up * x)
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up

        # Project back to embedding dimension
        y = self.w_down(hidden)

        if self.dropout is not None:
            y = self.dropout(y)

        return y
```

### Step 5: Update/Create Config Dataclass

Update `config/component_config.py` to add the new config class.

**Read the existing file first:**
```bash
# Check what configs exist
grep -E "^class.*Config:" packages/llmkit/llmkit/config/component_config.py
```

**Use the `Edit` tool to add the new config class:**

**Template for config class:**

```python
@dataclass
class {CONFIG_CLASS}:
    """Configuration for {COMPONENT_NAME}.

    Args:
        {CONFIG_DOCSTRING_PARAMS}
    """
    {CONFIG_FIELDS}

    def __post_init__(self):
        # Validation logic
        {VALIDATION_CODE}
```

**Example for MultiQueryAttention:**

```python
@dataclass
class MultiQueryAttentionConfig:
    """Configuration for multi-query attention.

    Args:
        n_embd: Embedding dimension
        n_head: Number of query heads
        num_kv_heads: Number of key-value heads (must divide n_head)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True)
        flash_attention: Whether to use flash attention (default: True)
    """
    n_embd: int
    n_head: int
    num_kv_heads: int = 1  # 1 = multi-query, n_head = standard MHA
    dropout: float = 0.0
    bias: bool = True
    flash_attention: bool = True

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_head % self.num_kv_heads != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by num_kv_heads ({self.num_kv_heads})")
```

**Example for SwiGLU:**

```python
@dataclass
class SwiGLUConfig:
    """Configuration for SwiGLU feedforward.

    Args:
        n_embd: Embedding dimension
        expansion_factor: Hidden dimension expansion factor (default: 4)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True)
    """
    n_embd: int
    expansion_factor: int = 4
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self):
        if self.expansion_factor < 1:
            raise ValueError(f"expansion_factor must be >= 1, got {self.expansion_factor}")
```

### Step 6: Update Package Exports

Update `core/__init__.py` to export the new component.

**Read current exports:**
```bash
cat packages/llmkit/llmkit/core/__init__.py
```

**Use the `Edit` tool to add new import and export:**

```python
# Add to imports
from .{snake_case_name} import {COMPONENT_NAME}

# Add to __all__
__all__ = [
    # ... existing exports ...
    "{COMPONENT_NAME}",
]
```

**Also update main `__init__.py` if component should be part of public API:**

```bash
# Check if component should be exported from main package
# Usually: Attention, Feedforward, Position Embeddings yes; helpers no
```

### Step 7: Generate Test Suite

Use the `Write` tool to create comprehensive tests.

**Template for `tests/unit/test_{snake_case_name}.py`:**

```python
"""Tests for {COMPONENT_NAME} component."""
import pytest
import torch
from llmkit.config.component_config import {CONFIG_CLASS}
from llmkit.core.{snake_case_name} import {COMPONENT_NAME}


class Test{COMPONENT_NAME}:
    """Tests for {COMPONENT_NAME} component."""

    def test_initialization(self):
        """Test component initializes correctly."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            {REQUIRED_PARAMS}
        )
        component = {COMPONENT_NAME}(config, layer_idx=0, n_layer=12)
        assert component is not None
        assert isinstance(component, torch.nn.Module)

    def test_forward_shape_preservation(self):
        """Test forward pass preserves input shape."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            {REQUIRED_PARAMS}
        )
        component = {COMPONENT_NAME}(config)

        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 128)
        y = component(x)

        assert y.shape == (batch_size, seq_len, 128)

    def test_config_validation(self):
        """Test config validation catches invalid parameters."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            {CONFIG_CLASS}(
                n_embd=127,  # Not divisible by n_head
                {INVALID_PARAMS}
            )

    def test_different_layer_indices(self):
        """Test component works with different layer indices."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})

        # Test first layer
        comp_first = {COMPONENT_NAME}(config, layer_idx=0, n_layer=12)

        # Test middle layer
        comp_middle = {COMPONENT_NAME}(config, layer_idx=6, n_layer=12)

        x = torch.randn(2, 16, 128)
        y_first = comp_first(x)
        y_middle = comp_middle(x)

        assert y_first.shape == y_middle.shape

    def test_gradient_flow(self):
        """Test gradients flow through component."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {COMPONENT_NAME}(config)

        x = torch.randn(2, 16, 128, requires_grad=True)
        y = component(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.parametrize("{PARAM_NAME}", {PARAM_VALUES})
    def test_parameter_variations(self, {PARAM_NAME}):
        """Test component with different parameter values."""
        config = {CONFIG_CLASS}(
            n_embd=128,
            {PARAM_NAME}={PARAM_NAME},
            {OTHER_REQUIRED_PARAMS}
        )
        component = {COMPONENT_NAME}(config)

        x = torch.randn(2, 16, 128)
        y = component(x)

        assert y.shape == x.shape

    def test_deterministic_output(self):
        """Test component produces deterministic output in eval mode."""
        config = {CONFIG_CLASS}(n_embd=128, {REQUIRED_PARAMS})
        component = {COMPONENT_NAME}(config)
        component.eval()

        x = torch.randn(2, 16, 128)
        y1 = component(x)
        y2 = component(x)

        assert torch.allclose(y1, y2)

    def test_parameter_count(self):
        """Test component has expected number of parameters."""
        config = {CONFIG_CLASS}(n_embd=768, {REQUIRED_PARAMS})
        component = {COMPONENT_NAME}(config)

        param_count = sum(p.numel() for p in component.parameters())

        # Expected parameter count calculation
        expected = {EXPECTED_PARAM_COUNT}

        assert param_count == expected, f"Expected {expected} parameters, got {param_count}"
```

**Specific test examples based on component type:**

**For Multi-Query Attention:**
```python
def test_kv_head_sharing(self):
    """Test that KV heads are correctly shared across query heads."""
    config = MultiQueryAttentionConfig(
        n_embd=768,
        n_head=12,
        num_kv_heads=4,  # 3 query heads per KV head
    )
    mqa = MultiQueryAttention(config)

    # K projection should have num_kv_heads * head_dim parameters
    head_dim = 768 // 12
    expected_k_params = 768 * (4 * head_dim)
    actual_k_params = mqa.k_proj.weight.numel()

    assert actual_k_params == expected_k_params

@pytest.mark.parametrize("num_kv_heads", [1, 2, 4, 12])
def test_different_kv_head_counts(self, num_kv_heads):
    """Test MQA with different numbers of KV heads."""
    config = MultiQueryAttentionConfig(
        n_embd=768,
        n_head=12,
        num_kv_heads=num_kv_heads,
    )
    mqa = MultiQueryAttention(config)

    x = torch.randn(2, 16, 768)
    y = mqa(x)

    assert y.shape == x.shape
```

**For SwiGLU:**
```python
def test_swiglu_activation(self):
    """Test SwiGLU applies gated activation correctly."""
    config = SwiGLUConfig(n_embd=128, expansion_factor=4)
    swiglu = SwiGLU(config)

    # SwiGLU should have gate and up projections
    assert hasattr(swiglu, 'w_gate')
    assert hasattr(swiglu, 'w_up')
    assert hasattr(swiglu, 'w_down')

    x = torch.randn(2, 16, 128)
    y = swiglu(x)

    assert y.shape == x.shape

@pytest.mark.parametrize("expansion_factor", [2, 4, 8])
def test_expansion_factors(self, expansion_factor):
    """Test SwiGLU with different expansion factors."""
    config = SwiGLUConfig(n_embd=128, expansion_factor=expansion_factor)
    swiglu = SwiGLU(config)

    # Check hidden dimension
    assert swiglu.w_gate.out_features == 128 * expansion_factor

    x = torch.randn(2, 16, 128)
    y = swiglu(x)

    assert y.shape == x.shape
```

### Step 8: Automatic Validation

Run validation checks to ensure everything works:

```bash
cd packages/llmkit

# 1. Import validation
echo "Validating imports..."
uv run python -c "
from llmkit.core import {COMPONENT_NAME}
from llmkit.config import {CONFIG_CLASS}
print('✓ Imports successful')
"

# 2. Config instantiation
echo "Validating config..."
uv run python -c "
from llmkit.config import {CONFIG_CLASS}
config = {CONFIG_CLASS}({CONFIG_INSTANTIATION_PARAMS})
print(f'✓ Config created: {config}')
"

# 3. Component instantiation
echo "Validating component..."
uv run python -c "
from llmkit.core import {COMPONENT_NAME}
from llmkit.config import {CONFIG_CLASS}
import torch

config = {CONFIG_CLASS}({CONFIG_INSTANTIATION_PARAMS})
component = {COMPONENT_NAME}(config)
x = torch.randn(2, 16, config.n_embd)
y = component(x)
assert y.shape == x.shape
print(f'✓ Component forward pass: {x.shape} -> {y.shape}')
"

# 4. Test execution
echo "Running tests..."
uv run pytest tests/unit/test_{snake_case_name}.py -v

# 5. Linting
echo "Running linter..."
uv run ruff check llmkit/core/{snake_case_name}.py

# 6. Full test suite (ensure no regressions)
echo "Running full test suite..."
uv run pytest tests/ -v

echo ""
echo "✅ All validations passed!"
```

### Step 9: Stage Files with Git

Stage the created/modified files (don't commit):

```bash
cd /Users/jex/frontier-ready

# Stage all modified files
git add packages/llmkit/llmkit/core/{snake_case_name}.py
git add packages/llmkit/llmkit/config/component_config.py
git add packages/llmkit/llmkit/core/__init__.py
git add packages/llmkit/tests/unit/test_{snake_case_name}.py

# Show what's staged
git status
```

### Step 10: Summary Output

Print a comprehensive summary for the user:

```
✅ Component {COMPONENT_NAME} created successfully!

📁 Files created/modified:
  ✓ packages/llmkit/llmkit/core/{snake_case_name}.py
  ✓ packages/llmkit/llmkit/config/component_config.py
  ✓ packages/llmkit/llmkit/core/__init__.py
  ✓ packages/llmkit/tests/unit/test_{snake_case_name}.py

✓ Validations passed:
  ✓ Import check
  ✓ Config instantiation
  ✓ Forward pass shape preservation
  ✓ Test suite execution ({num_tests} tests)
  ✓ Linting (ruff)

📝 Suggested commit message:

Add {COMPONENT_TYPE}: {COMPONENT_NAME}

- {DESCRIPTION}
- Config: {CONFIG_CLASS} with {config_params}
- Tests: {num_tests} test cases covering initialization, forward, validation
- Integration: Exported from llmkit.core

{ADDITIONAL_NOTES}

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

📦 Next steps:
  1. Review generated code: packages/llmkit/llmkit/core/{snake_case_name}.py
  2. Verify tests pass: uv run pytest tests/unit/test_{snake_case_name}.py -v
  3. Commit changes: git commit -m "Add {COMPONENT_TYPE}: {COMPONENT_NAME}"
  4. (Optional) Create variant: /add-variant using this component
  5. (Optional) Create experiment: /new-experiment to test component

💡 Usage example:

```python
from llmkit import GPTConfig
from llmkit.config import {CONFIG_CLASS}
from llmkit.core import {COMPONENT_NAME}

# Create config
config = {CONFIG_CLASS}({EXAMPLE_CONFIG_PARAMS})

# Instantiate component
component = {COMPONENT_NAME}(config, layer_idx=0, n_layer=12)

# Use in forward pass
import torch
x = torch.randn(batch_size, seq_len, config.n_embd)
output = component(x)
```
```

## Error Handling

### File Already Exists
```bash
if [ -f "${COMPONENT_FILE}" ]; then
    echo "⚠️  Component ${COMPONENT_NAME} already exists"
    # Use AskUserQuestion to ask if they want to overwrite
    # Options: "Overwrite", "Choose different name", "Cancel"
fi
```

### Invalid Component Name
```bash
# Must be CamelCase
if ! echo "${COMPONENT_NAME}" | grep -qE '^[A-Z][a-zA-Z0-9]*$'; then
    echo "❌ Component name must be CamelCase (e.g., 'MultiQueryAttention')"
    exit 1
fi
```

### Import Errors
```bash
# If imports fail during validation
if ! uv run python -c "from llmkit.core import ${COMPONENT_NAME}" 2>/dev/null; then
    echo "❌ Import failed. Possible issues:"
    echo "  - Syntax error in component file"
    echo "  - Missing dependency"
    echo "  - Incorrect __init__.py exports"
    echo ""
    echo "Run: uv run python -c 'from llmkit.core import ${COMPONENT_NAME}' to see error"
    exit 1
fi
```

### Test Failures
```bash
# If tests fail
if ! uv run pytest tests/unit/test_${snake_case_name}.py -v; then
    echo "⚠️  Some tests failed. Review test output above."
    echo "Common issues:"
    echo "  - Shape mismatch in forward pass"
    echo "  - Missing required config parameters"
    echo "  - Incorrect parameter initialization"
    echo ""
    echo "You can still commit and fix tests later, or fix now."
fi
```

### Workspace Sync Errors
```bash
# If uv sync fails
if ! uv sync 2>/dev/null; then
    echo "⚠️  Workspace sync failed"
    echo "Run manually: uv sync"
    echo "This may indicate dependency issues"
fi
```

## Notes

- **Layer-aware initialization:** All components should accept `layer_idx` and `n_layer` for residual scaling
- **Config integration:** Always create a config dataclass, never pass individual parameters
- **ScaledLinear:** Use for output projections with scale_factor = (2 * n_layer) ** -0.5
- **Docstrings:** Google-style docstrings for all classes and methods
- **Type hints:** Use torch.Tensor type hints for forward signatures
- **Dropout:** Optional dropout in config, conditional initialization (only create if > 0)
- **Bias:** Configurable bias (default True for GPT-2 compatibility)
- **Flash attention:** Support flash attention for attention components when possible
- **Tests:** Aim for >90% coverage with initialization, shape, validation, parameter, gradient tests
- **Naming:** CamelCase for classes, snake_case for files/modules
- **Validation:** Config __post_init__ should validate parameter constraints

## Reference Files

- **Format reference:** `.claude/skills/new-experiment/skill.md`
- **Component example:** `packages/llmkit/llmkit/core/attention.py`
- **Config example:** `packages/llmkit/llmkit/config/component_config.py`
- **Test example:** `packages/llmkit/tests/unit/test_feedforward.py`
- **Init scaling:** `packages/llmkit/llmkit/core/initialization.py`
