# Add Variant Skill

Creates architecture variant in `llmkit/variants/` with factory function, tests, and documentation.

## Usage

```
/add-variant
```

## What This Skill Does

1. Prompts for variant details (name, base model, modifications)
2. Creates variant module in `llmkit/variants/`
3. Generates factory function (`create_{name}_gpt`)
4. Updates package exports
5. Generates variant tests
6. Updates README.md variants section
7. Runs automatic validation (import, model creation, forward pass)
8. Stages files with git (user handles commit)

## Execution Steps

### Step 1: Gather Information

Use the `AskUserQuestion` tool to collect variant details:

**Questions to ask:**

1. **Variant Name** (header: "Name")
   - Question: "What is the variant name? (lowercase, e.g., 'rope', 'swiglu', 'gqa', 'parallel')"
   - Options:
     - label: "rope", description: "Rotary position embeddings"
     - label: "swiglu", description: "SwiGLU activation in feedforward"
     - label: "gqa", description: "Grouped-query attention"
     - label: "parallel", description: "Parallel attention and MLP"

2. **Description** (header: "Description")
   - Question: "One-sentence description of what this variant does?"
   - Options:
     - label: "Replaces learned position embeddings with rotary embeddings", description: "Position embedding change"
     - label: "Uses SwiGLU activation instead of GELU", description: "Activation function change"
     - label: "Reduces KV cache with grouped-query attention", description: "Attention mechanism change"
     - label: "Parallel attention and MLP for efficiency", description: "Block structure change"

3. **Base Model** (header: "Base")
   - Question: "What model does this variant modify?"
   - Options:
     - label: "GPT", description: "Standard GPT model (most common)"
     - label: "Custom", description: "Custom model (not GPT)"

4. **Modifications** (header: "Modifications", multiSelect: true)
   - Question: "What components does this variant modify?"
   - Options:
     - label: "Position embeddings", description: "Changes position_embedding_type or position_embedding_config"
     - label: "Attention mechanism", description: "Changes attention implementation or config"
     - label: "Feedforward/MLP", description: "Changes feedforward implementation or config"
     - label: "Normalization", description: "Changes normalization layers"
     - label: "Block structure", description: "Changes how transformer blocks are organized"
   - multiSelect: true

5. **Config Changes** (header: "Config")
   - Question: "What config parameters need to be set? (e.g., 'position_embedding_type=rope, rope_theta=10000')"
   - Options:
     - label: "position_embedding_type=rope", description: "Set position embedding type to RoPE"
     - label: "activation=silu", description: "Change activation to SiLU"
     - label: "num_kv_heads=4", description: "Set number of KV heads for GQA"
     - label: "Custom", description: "Enter custom config changes"

### Step 2: Validate Variant Name

Validate the variant name before proceeding:

```bash
VARIANT_NAME="{VARIANT_NAME}"  # e.g., "rope"

# Check if variant already exists
if [ -f "packages/llmkit/llmkit/variants/${VARIANT_NAME}.py" ]; then
    echo "⚠️  Variant ${VARIANT_NAME} already exists"
    # Ask user if they want to overwrite
fi

# Validate naming (lowercase, no special chars except underscore)
if ! echo "${VARIANT_NAME}" | grep -qE '^[a-z][a-z0-9_]*$'; then
    echo "❌ Variant name must be lowercase with underscores (e.g., 'rope', 'swiglu', 'multi_query')"
    exit 1
fi

echo "✓ Variant name valid: ${VARIANT_NAME}"
```

### Step 3: Determine File Locations

Based on the variant name, determine all file paths:

```bash
VARIANT_NAME="{VARIANT_NAME}"  # e.g., "rope"
FUNCTION_NAME="create_${VARIANT_NAME}_gpt"  # e.g., "create_rope_gpt"

# File paths
VARIANT_FILE="packages/llmkit/llmkit/variants/${VARIANT_NAME}.py"
VARIANTS_INIT="packages/llmkit/llmkit/variants/__init__.py"
TEST_FILE="packages/llmkit/tests/unit/test_variant_${VARIANT_NAME}.py"
README_FILE="packages/llmkit/README.md"

echo "📁 Files to create/modify:"
echo "  - Variant: ${VARIANT_FILE}"
echo "  - Exports: ${VARIANTS_INIT} (update)"
echo "  - Tests: ${TEST_FILE}"
echo "  - Docs: ${README_FILE} (update)"
```

### Step 4: Create Variant Module

Use the `Write` tool to create the variant module.

**Template variables:**
- `{VARIANT_NAME}`: Lowercase variant name (e.g., "rope")
- `{FUNCTION_NAME}`: Factory function name (e.g., "create_rope_gpt")
- `{DESCRIPTION}`: One-sentence description
- `{CONFIG_CHANGES}`: Config modifications as Python code
- `{BASE_MODEL}`: GPT or custom model class

**Template for `variants/{variant_name}.py`:**

```python
"""{DESCRIPTION}

This variant modifies the standard GPT architecture by:
{MODIFICATION_DETAILS}

Usage example:
    from llmkit.variants.{VARIANT_NAME} import {FUNCTION_NAME}
    from llmkit import ModelConfig

    config = ModelConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        vocab_size=50257,
    )
    model = {FUNCTION_NAME}(config)

    # Or configure manually:
    from llmkit import GPT, ModelConfig
    config = ModelConfig({CONFIG_EXAMPLE})
    model = GPT(config)
"""
from ..models import {BASE_MODEL}
from ..config import ModelConfig
{COMPONENT_IMPORTS}


def {FUNCTION_NAME}(config: ModelConfig) -> {BASE_MODEL}:
    """Create a {BASE_MODEL} model with {DESCRIPTION}.

    Args:
        config: Model configuration

    Returns:
        {BASE_MODEL} model with {VARIANT_NAME} modifications
    """
    # Modify config for variant
{CONFIG_MODIFICATIONS}

    return {BASE_MODEL}(config)
```

**Specific examples based on variant type:**

**For RoPE (position embedding variant):**
```python
"""RoPE (Rotary Position Embeddings) variant.

This variant modifies the standard GPT architecture by:
- Replacing learned positional embeddings (wpe) with rotary embeddings
- Applying rotary embeddings to Q and K vectors in attention

Usage example:
    from llmkit.variants.rope import create_rope_gpt
    from llmkit import ModelConfig

    config = ModelConfig()
    model = create_rope_gpt(config)

    # Or use directly:
    from llmkit import GPT, ModelConfig
    config = ModelConfig(position_embedding_type="rope")
    model = GPT(config)
"""
from ..models import GPT
from ..config import ModelConfig
from ..core.embedding import RoPEPositionEmbedding


def create_rope_gpt(config: ModelConfig) -> GPT:
    """Create a GPT model with RoPE position embeddings.

    Args:
        config: Model configuration

    Returns:
        GPT model with RoPE position embeddings
    """
    # Set position embedding type to RoPE
    config.position_embedding_type = "rope"

    return GPT(config)
```

**For SwiGLU (feedforward variant):**
```python
"""SwiGLU activation variant.

This variant modifies the standard GPT architecture by:
- Replacing GELU activation with SwiGLU in the feedforward network
- Using gated linear units for improved expressiveness

Usage example:
    from llmkit.variants.swiglu import create_swiglu_gpt
    from llmkit import ModelConfig

    config = ModelConfig()
    model = create_swiglu_gpt(config)

    # Or use directly:
    from llmkit import GPT, ModelConfig
    from llmkit.config import FeedForwardConfig
    config = ModelConfig(
        feedforward_config=FeedForwardConfig(
            n_embd=768,
            activation="swiglu",
        )
    )
    model = GPT(config)
"""
from ..models import GPT
from ..config import ModelConfig, FeedForwardConfig


def create_swiglu_gpt(config: ModelConfig) -> GPT:
    """Create a GPT model with SwiGLU activation.

    Args:
        config: Model configuration

    Returns:
        GPT model with SwiGLU activation in feedforward
    """
    # Modify feedforward config to use SwiGLU
    if config.feedforward_config is None:
        config.feedforward_config = FeedForwardConfig(
            n_embd=config.n_embd,
            activation="swiglu",
        )
    else:
        config.feedforward_config.activation = "swiglu"

    return GPT(config)
```

**For GQA (attention variant):**
```python
"""Grouped-Query Attention (GQA) variant.

This variant modifies the standard GPT architecture by:
- Using grouped-query attention to reduce KV cache size
- Sharing key-value heads across multiple query heads

Usage example:
    from llmkit.variants.gqa import create_gqa_gpt
    from llmkit import ModelConfig

    config = ModelConfig(n_head=12)
    model = create_gqa_gpt(config, num_kv_heads=4)  # 3 query heads per KV head

    # Or use directly:
    from llmkit import GPT, ModelConfig
    from llmkit.config import AttentionConfig
    config = ModelConfig(
        attention_config=AttentionConfig(
            n_embd=768,
            n_head=12,
            num_kv_heads=4,
        )
    )
    model = GPT(config)
"""
from ..models import GPT
from ..config import ModelConfig, AttentionConfig


def create_gqa_gpt(config: ModelConfig, num_kv_heads: int = 1) -> GPT:
    """Create a GPT model with grouped-query attention.

    Args:
        config: Model configuration
        num_kv_heads: Number of key-value heads (default: 1 for multi-query attention)

    Returns:
        GPT model with grouped-query attention
    """
    # Modify attention config to use GQA
    if config.attention_config is None:
        config.attention_config = AttentionConfig(
            n_embd=config.n_embd,
            n_head=config.n_head,
            num_kv_heads=num_kv_heads,
        )
    else:
        config.attention_config.num_kv_heads = num_kv_heads

    return GPT(config)
```

### Step 5: Update Package Exports

Update `variants/__init__.py` to export the new variant:

```bash
# Read current exports
cat packages/llmkit/llmkit/variants/__init__.py
```

**Use the `Edit` tool to add new import and export:**

```python
# Add to imports
from .{variant_name} import {FUNCTION_NAME}

# Add to __all__
__all__ = [
    # ... existing exports ...
    "{FUNCTION_NAME}",
]
```

### Step 6: Generate Variant Tests

Use the `Write` tool to create tests for the variant.

**Template for `tests/unit/test_variant_{variant_name}.py`:**

```python
"""Tests for {VARIANT_NAME} variant."""
import pytest
import torch
from llmkit import ModelConfig, GPT
from llmkit.variants.{VARIANT_NAME} import {FUNCTION_NAME}


class TestVariant{VARIANT_NAME_CAMEL}:
    """Tests for {VARIANT_NAME} variant."""

    def test_variant_creation(self):
        """Test variant creates model successfully."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        model = {FUNCTION_NAME}(config)

        assert isinstance(model, GPT)
        assert model is not None

    def test_variant_forward(self):
        """Test variant model forward pass."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        model = {FUNCTION_NAME}(config)
        model.eval()

        x = torch.randint(0, 100, (2, 16))
        logits, loss = model(x, x)

        assert logits.shape == (2, 16, 100)
        assert loss is not None

    def test_variant_config_modification(self):
        """Test variant properly modifies config."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        model = {FUNCTION_NAME}(config)

        # Check that config was modified as expected
{CONFIG_ASSERTIONS}

    def test_variant_parameter_count(self):
        """Test variant has expected parameter count."""
        config = ModelConfig(
            vocab_size=50257,
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=1024,
        )

        baseline = GPT(config)
        variant = {FUNCTION_NAME}(config)

        baseline_params = sum(p.numel() for p in baseline.parameters())
        variant_params = sum(p.numel() for p in variant.parameters())

        # Variant may have different parameter count
        # (e.g., RoPE removes wpe, GQA reduces KV projections)
{PARAM_COUNT_ASSERTIONS}

    def test_variant_training_step(self):
        """Test variant can perform training step."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        model = {FUNCTION_NAME}(config)
        model.train()

        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))

        logits, loss = model(x, y)

        # Backward pass
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_variant_deterministic_eval(self):
        """Test variant produces deterministic output in eval mode."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        model = {FUNCTION_NAME}(config)
        model.eval()

        x = torch.randint(0, 100, (2, 16))

        with torch.no_grad():
            logits1, _ = model(x)
            logits2, _ = model(x)

        assert torch.allclose(logits1, logits2)

{VARIANT_SPECIFIC_TESTS}
```

**Variant-specific test examples:**

**For RoPE:**
```python
    def test_rope_no_learned_position_embeddings(self):
        """Test RoPE variant doesn't have learned position embeddings."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        model = create_rope_gpt(config)

        # RoPE models shouldn't have wpe (learned position embeddings)
        assert not hasattr(model.transformer, 'wpe') or model.transformer.wpe is None
```

**For SwiGLU:**
```python
    def test_swiglu_activation_used(self):
        """Test SwiGLU variant uses SwiGLU activation."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=4,
            n_embd=128,
            block_size=32,
        )

        model = create_swiglu_gpt(config)

        # Check that blocks use SwiGLU
        # (Implementation depends on how SwiGLU is structured)
        assert config.feedforward_config.activation == "swiglu"
```

**For GQA:**
```python
    def test_gqa_kv_head_count(self):
        """Test GQA variant has correct number of KV heads."""
        config = ModelConfig(
            vocab_size=100,
            n_layer=2,
            n_head=12,
            n_embd=768,
            block_size=32,
        )

        model = create_gqa_gpt(config, num_kv_heads=4)

        # Verify KV head configuration
        assert config.attention_config.num_kv_heads == 4
        assert config.attention_config.n_head == 12
```

### Step 7: Update README Documentation

Update `packages/llmkit/README.md` to document the new variant:

**Read the README to find the variants section:**
```bash
grep -n "## Variants" packages/llmkit/README.md
# Or search for architectural variants section
```

**Use the `Edit` tool to add variant documentation:**

```markdown
### {VARIANT_NAME}

{DESCRIPTION}

**Usage:**
```python
from llmkit.variants.{VARIANT_NAME} import {FUNCTION_NAME}
from llmkit import ModelConfig

config = ModelConfig()
model = {FUNCTION_NAME}(config)
```

**What it does:**
{DETAILED_EXPLANATION}

**Parameter differences:**
- {PARAM_DIFF_1}
- {PARAM_DIFF_2}

**References:**
- {PAPER_REFERENCE}
```

### Step 8: Automatic Validation

Run validation checks:

```bash
cd packages/llmkit

# 1. Import validation
echo "Validating imports..."
uv run python -c "
from llmkit.variants.{VARIANT_NAME} import {FUNCTION_NAME}
from llmkit import ModelConfig
print('✓ Imports successful')
"

# 2. Model creation
echo "Validating model creation..."
uv run python -c "
from llmkit.variants.{VARIANT_NAME} import {FUNCTION_NAME}
from llmkit import ModelConfig
import torch

config = ModelConfig(
    vocab_size=100,
    n_layer=2,
    n_head=4,
    n_embd=128,
    block_size=32,
)
model = {FUNCTION_NAME}(config)
print(f'✓ Model created: {type(model).__name__}')
"

# 3. Forward pass
echo "Validating forward pass..."
uv run python -c "
from llmkit.variants.{VARIANT_NAME} import {FUNCTION_NAME}
from llmkit import ModelConfig
import torch

config = ModelConfig(
    vocab_size=100,
    n_layer=2,
    n_head=4,
    n_embd=128,
    block_size=32,
)
model = {FUNCTION_NAME}(config)
model.eval()

x = torch.randint(0, 100, (2, 16))
with torch.no_grad():
    logits, _ = model(x)
print(f'✓ Forward pass: {x.shape} -> {logits.shape}')
"

# 4. Test execution
echo "Running variant tests..."
uv run pytest tests/unit/test_variant_{VARIANT_NAME}.py -v

# 5. Full test suite (ensure no regressions)
echo "Running full test suite..."
uv run pytest tests/ -v

echo ""
echo "✅ All validations passed!"
```

### Step 9: Stage Files with Git

Stage the created/modified files:

```bash
cd /Users/jex/frontier-ready

# Stage variant module
git add packages/llmkit/llmkit/variants/{VARIANT_NAME}.py
git add packages/llmkit/llmkit/variants/__init__.py

# Stage tests
git add packages/llmkit/tests/unit/test_variant_{VARIANT_NAME}.py

# Stage docs
git add packages/llmkit/README.md

# Show what's staged
git status
```

### Step 10: Summary Output

Print comprehensive summary:

```
✅ Variant {VARIANT_NAME} created successfully!

📁 Files created/modified:
  ✓ packages/llmkit/llmkit/variants/{VARIANT_NAME}.py
  ✓ packages/llmkit/llmkit/variants/__init__.py
  ✓ packages/llmkit/tests/unit/test_variant_{VARIANT_NAME}.py
  ✓ packages/llmkit/README.md

✓ Validations passed:
  ✓ Import check
  ✓ Model creation
  ✓ Forward pass
  ✓ Test suite ({num_tests} tests)
  ✓ Full test suite: no regressions

📝 Suggested commit message:

Add architecture variant: {VARIANT_NAME}

- {DESCRIPTION}
- Factory function: {FUNCTION_NAME}
- Modifications: {MODIFICATIONS_LIST}
- Tests: {num_tests} test cases

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

📦 Next steps:
  1. Review variant: packages/llmkit/llmkit/variants/{VARIANT_NAME}.py
  2. Run tests: uv run pytest tests/unit/test_variant_{VARIANT_NAME}.py -v
  3. Commit: git commit -m "Add variant: {VARIANT_NAME}"
  4. Create experiment: /new-experiment (select "Architecture variant" and "{VARIANT_NAME}")

💡 Usage example:

```python
from llmkit.variants.{VARIANT_NAME} import {FUNCTION_NAME}
from llmkit import ModelConfig

config = ModelConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    vocab_size=50257,
)

model = {FUNCTION_NAME}(config)

# Use in training
from llmkit import Trainer, TrainingConfig

training_config = TrainingConfig(
    model_config=config,
    # ... other training params
)

trainer = Trainer(training_config)
trainer.train()
```

🔬 Parameter comparison:
  Baseline: {baseline_params:,} parameters
  {VARIANT_NAME}: {variant_params:,} parameters
  Difference: {param_diff:+,} ({percent_diff:+.1f}%)
```

## Error Handling

### Variant Already Exists
```bash
if [ -f "${VARIANT_FILE}" ]; then
    echo "⚠️  Variant ${VARIANT_NAME} already exists"
    # Use AskUserQuestion
    # Options: "Overwrite", "Choose different name", "Cancel"
fi
```

### Invalid Variant Name
```bash
if ! echo "${VARIANT_NAME}" | grep -qE '^[a-z][a-z0-9_]*$'; then
    echo "❌ Variant name must be lowercase with underscores"
    echo "Valid examples: rope, swiglu, multi_query"
    exit 1
fi
```

### Import Errors
```bash
if ! uv run python -c "from llmkit.variants.${VARIANT_NAME} import ${FUNCTION_NAME}" 2>/dev/null; then
    echo "❌ Import failed"
    echo "Possible issues:"
    echo "  - Syntax error in variant file"
    echo "  - Missing component dependency"
    echo "  - Incorrect __init__.py exports"
    exit 1
fi
```

### Model Creation Errors
```bash
if ! uv run python -c "
from llmkit.variants.${VARIANT_NAME} import ${FUNCTION_NAME}
from llmkit import ModelConfig
config = ModelConfig()
model = ${FUNCTION_NAME}(config)
" 2>/dev/null; then
    echo "❌ Model creation failed"
    echo "Possible issues:"
    echo "  - Config modification error"
    echo "  - Component not available"
    echo "  - Invalid config parameters"
    exit 1
fi
```

### Test Failures
```bash
if ! uv run pytest tests/unit/test_variant_${VARIANT_NAME}.py -v; then
    echo "⚠️  Some tests failed"
    echo "Review test output above and fix failing tests"
    echo "Tests are staged - you can commit and fix later"
fi
```

## Notes

- **Factory pattern:** Use `create_{name}_gpt()` naming convention
- **Config modification:** Modify config before passing to GPT constructor
- **Thin wrappers:** Variants should be simple config modifications, not complex logic
- **Docstrings:** Include usage examples in module docstring
- **Tests:** Verify config modification, forward pass, parameter count
- **Documentation:** Update README with variant description and usage
- **Comparison:** Test variant against baseline for parameter differences
- **Integration:** Variant should work with standard Trainer and evaluation
- **Naming:** Lowercase with underscores (rope, swiglu, multi_query)
- **Dependencies:** Ensure required components exist before creating variant

## Reference Files

- **Variant example:** `packages/llmkit/llmkit/variants/rope.py`
- **Variant init:** `packages/llmkit/llmkit/variants/__init__.py`
- **Model config:** `packages/llmkit/llmkit/config/model_config.py`
- **GPT model:** `packages/llmkit/llmkit/models/gpt.py`
