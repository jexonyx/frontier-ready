# Update Documentation Skill

Updates package and workspace documentation after adding components, variants, or packages.

## Usage

```
/update-docs
```

## What This Skill Does

1. Prompts for update scope (component/variant/package/workspace)
2. Detects recent changes via git diff
3. Analyzes changes to identify new classes/functions
4. Determines documentation files to update
5. Updates package README with features, examples, API reference
6. Updates workspace README if needed
7. Generates runnable code examples
8. Updates/creates CHANGELOG.md
9. Runs automatic validation (markdown syntax, example execution, links)
10. Stages files with git (user handles commit)

## Execution Steps

### Step 1: Gather Information

Use the `AskUserQuestion` tool to collect update requirements:

**Questions to ask:**

1. **Update Scope** (header: "Scope")
   - Question: "What documentation needs updating?"
   - Options:
     - label: "Component", description: "Document new component in llmkit/core/"
     - label: "Variant", description: "Document architecture variant in llmkit/variants/"
     - label: "Package", description: "Document entire package (new or updated)"
     - label: "Workspace", description: "Update workspace-level README"

2. **What Changed** (header: "Changes")
   - Question: "Describe what changed (or select 'Auto-detect from git')"
   - Options:
     - label: "Auto-detect", description: "Analyze git diff to find changes"
     - label: "New component added", description: "Added component to core/"
     - label: "New variant added", description: "Added variant to variants/"
     - label: "Package structure changed", description: "Major package refactoring"

3. **Include Examples** (header: "Examples")
   - Question: "How should code examples be handled?"
   - Options:
     - label: "Generate examples", description: "Auto-generate usage examples from code"
     - label: "I'll provide examples", description: "User will write examples"
     - label: "Minimal examples only", description: "Just basic import/usage"

4. **API Documentation** (header: "API Docs")
   - Question: "How detailed should API documentation be?"
   - Options:
     - label: "Full", description: "Complete API reference with all classes/functions"
     - label: "Brief", description: "Just main classes and key functions"
     - label: "None", description: "Skip API documentation"

### Step 2: Detect Recent Changes

Analyze git to find what changed:

```bash
# Check for uncommitted changes
UNCOMMITTED=$(git diff --name-only)
STAGED=$(git diff --staged --name-only)

if [ -n "${STAGED}" ]; then
    echo "Found staged changes:"
    echo "${STAGED}"
    DIFF_FILES="${STAGED}"
elif [ -n "${UNCOMMITTED}" ]; then
    echo "Found uncommitted changes:"
    echo "${UNCOMMITTED}"
    DIFF_FILES="${UNCOMMITTED}"
else
    # Check last commit
    echo "Checking last commit..."
    DIFF_FILES=$(git diff HEAD~1..HEAD --name-only)
fi

# Filter for Python files
PYTHON_FILES=$(echo "${DIFF_FILES}" | grep "\.py$")

echo "Python files changed:"
echo "${PYTHON_FILES}"
```

### Step 3: Analyze Changes

Parse changed Python files to extract new classes and functions:

```bash
for file in ${PYTHON_FILES}; do
    echo "Analyzing ${file}..."

    # Extract class definitions
    CLASSES=$(grep "^class " "${file}" | sed 's/class \([^(:]*\).*/\1/')

    # Extract function definitions (top-level only)
    FUNCTIONS=$(grep "^def " "${file}" | sed 's/def \([^(]*\).*/\1/')

    # Extract docstrings
    DOCSTRINGS=$(grep -A 1 '"""' "${file}" | head -1)

    echo "  Classes: ${CLASSES}"
    echo "  Functions: ${FUNCTIONS}"
done
```

### Step 4: Determine Documentation Files

Based on scope, determine which files need updating:

```bash
case "${SCOPE}" in
    "Component")
        DOC_FILES="packages/llmkit/README.md"
        ;;
    "Variant")
        DOC_FILES="packages/llmkit/README.md"
        ;;
    "Package")
        PACKAGE_NAME=$(echo "${PYTHON_FILES}" | head -1 | cut -d'/' -f2)
        DOC_FILES="packages/${PACKAGE_NAME}/README.md"
        ;;
    "Workspace")
        DOC_FILES="README.md"
        ;;
esac

echo "Documentation files to update:"
echo "${DOC_FILES}"
```

### Step 5: Update Package README

**For Component documentation:**

Add section to llmkit README.md:

```markdown
### {COMPONENT_NAME}

{DESCRIPTION}

**Usage:**
```python
from llmkit.core import {COMPONENT_NAME}
from llmkit.config import {CONFIG_CLASS}

config = {CONFIG_CLASS}(
    n_embd=768,
    n_head=12,
    {COMPONENT_SPECIFIC_PARAMS}
)

component = {COMPONENT_NAME}(config, layer_idx=0, n_layer=12)

# Use in forward pass
import torch
x = torch.randn(batch_size, seq_len, 768)
output = component(x)
```

**Configuration:**
- `{param1}` ({type}): {description}
- `{param2}` ({type}): {description}

**Key features:**
- {feature1}
- {feature2}
```

**For Variant documentation:**

Add section to llmkit README.md:

```markdown
### {VARIANT_NAME}

{DESCRIPTION}

**Usage:**
```python
from llmkit.variants.{VARIANT_NAME} import create_{VARIANT_NAME}_gpt
from llmkit import ModelConfig

config = ModelConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    block_size=1024,
    vocab_size=50257,
)

model = create_{VARIANT_NAME}_gpt(config)
```

**What it does:**
{DETAILED_EXPLANATION}

**Differences from baseline:**
- {difference1}
- {difference2}

**References:**
- {paper_link}
```

**For Package documentation:**

Update package README.md with:

```markdown
## Recent Updates

### Version {VERSION} ({DATE})

**New features:**
- {feature1}
- {feature2}

**Breaking changes:**
- {change1}

**Bug fixes:**
- {fix1}

## Installation

```bash
# From workspace root
uv sync
```

## Usage

```python
from {PACKAGE_NAME} import {NEW_CLASS}

# Example
{USAGE_EXAMPLE}
```

## API Reference

### {NEW_CLASS}

{CLASS_DESCRIPTION}

**Parameters:**
- `param1` ({type}): {description}

**Methods:**
- `method1()`: {description}

**Example:**
```python
{EXAMPLE_CODE}
```
```

### Step 6: Generate Code Examples

Create runnable code examples:

```python
# Example template
example_code = f"""
from llmkit.core import {component_name}
from llmkit.config import {config_class}
import torch

# Create configuration
config = {config_class}(
    n_embd=768,
    n_head=12,
)

# Initialize component
component = {component_name}(config)

# Run forward pass
x = torch.randn(2, 16, 768)
y = component(x)

print(f"Input shape: {{x.shape}}")
print(f"Output shape: {{y.shape}}")
"""

# Write example to file
with open(f"examples/{component_name}_example.py", "w") as f:
    f.write(example_code)
```

### Step 7: Update CHANGELOG.md

Create or update CHANGELOG.md:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- {COMPONENT_NAME}: {DESCRIPTION}
- {FEATURE_DESCRIPTION}

### Changed
- {CHANGE_DESCRIPTION}

### Fixed
- {FIX_DESCRIPTION}

## [{VERSION}] - {DATE}

### Added
- {PREVIOUS_FEATURE}
```

### Step 8: Automatic Validation

Validate documentation updates:

```bash
# 1. Markdown syntax check
echo "Checking markdown syntax..."
for doc in ${DOC_FILES}; do
    # Check for common markdown issues
    grep -n "^#[^# ]" "${doc}" && echo "⚠️  Missing space after # in ${doc}"
    grep -n "\`\`\`$" "${doc}" && echo "⚠️  Code block not closed in ${doc}"
done
echo "✓ Markdown syntax OK"

# 2. Validate code examples
echo "Validating code examples..."
for example in examples/*.py; do
    if [ -f "${example}" ]; then
        echo "  Testing ${example}..."
        uv run python "${example}"
        if [ $? -eq 0 ]; then
            echo "  ✓ ${example} runs successfully"
        else
            echo "  ⚠️  ${example} failed"
        fi
    fi
done

# 3. Link checking
echo "Checking links..."
for doc in ${DOC_FILES}; do
    # Extract markdown links
    LINKS=$(grep -o '\[.*\]([^)]*)' "${doc}" | sed 's/.*(\([^)]*\)).*/\1/')

    for link in ${LINKS}; do
        if [[ "${link}" == http* ]]; then
            # External link - skip for now
            continue
        else
            # Internal link - check file exists
            if [ ! -f "${link}" ]; then
                echo "⚠️  Broken link in ${doc}: ${link}"
            fi
        fi
    done
done
echo "✓ Links checked"

# 4. Consistency check
echo "Checking consistency..."
# Verify all exported items are documented
uv run python -c "
import ${PACKAGE_NAME}
exports = ${PACKAGE_NAME}.__all__

print(f'Checking {len(exports)} exports are documented...')
# Read README and check each export is mentioned
with open('${README_FILE}') as f:
    readme = f.read()

for item in exports:
    if item not in readme and item != '__version__':
        print(f'⚠️  {item} exported but not documented')
"

echo "✓ Documentation complete"
```

### Step 9: Stage Files with Git

```bash
cd /Users/jex/frontier-ready

# Stage documentation files
for doc in ${DOC_FILES}; do
    git add "${doc}"
done

# Stage CHANGELOG if created
if [ -f "CHANGELOG.md" ]; then
    git add CHANGELOG.md
fi

# Stage examples if created
if [ -d "examples" ]; then
    git add examples/
fi

git status
```

### Step 10: Summary Output

```
✅ Documentation updated successfully!

📁 Files updated:
  ✓ ${DOC_FILE_1}
  ✓ ${DOC_FILE_2}
  ✓ CHANGELOG.md
  ✓ examples/{EXAMPLE_FILES}

✓ Validations passed:
  ✓ Markdown syntax valid
  ✓ Code examples run successfully ({num_examples} examples)
  ✓ Links checked ({num_links} links)
  ✓ Consistency check passed

📝 Suggested commit message:

Update documentation for {SCOPE}

- Added documentation for {ITEMS}
- Updated README with usage examples
- Generated {num_examples} code examples
- Updated CHANGELOG

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

📦 Next steps:
  1. Review docs: ${DOC_FILES}
  2. Test examples: uv run python examples/{EXAMPLE}.py
  3. Commit: git commit -m "Update documentation"

💡 Documentation sections added:
  {SECTIONS_LIST}

📊 Coverage:
  Documented: {documented_items}/{total_items} exported items
  Examples: {num_examples}
  API reference: {api_detail_level}
```

## Error Handling

### No Changes Detected
```bash
if [ -z "${DIFF_FILES}" ]; then
    echo "⚠️  No changes detected"
    echo "Either:"
    echo "  1. Commit recent changes first"
    echo "  2. Or provide manual description of changes"
fi
```

### Markdown Syntax Errors
```bash
if grep -q "^#[^# ]" "${doc}"; then
    echo "⚠️  Markdown syntax issues found"
    echo "Fix: Add space after # in headers"
    echo "Example: '#Header' → '# Header'"
fi
```

### Example Validation Failures
```bash
if ! uv run python "${example}"; then
    echo "⚠️  Example ${example} failed to run"
    echo "Review example code for errors"
    echo "Example is still staged - fix before committing"
fi
```

### Broken Links
```bash
if [ ! -f "${link}" ]; then
    echo "⚠️  Broken link: ${link}"
    echo "Either:"
    echo "  1. Create the missing file"
    echo "  2. Update the link in documentation"
fi
```

## Notes

- **Auto-detect:** Use git diff to find recent changes
- **Code examples:** Generate runnable examples that demonstrate usage
- **API docs:** Document all public classes and functions
- **CHANGELOG:** Follow Keep a Changelog format
- **Markdown:** Use proper syntax (space after #, closed code blocks)
- **Links:** Validate internal links point to existing files
- **Consistency:** Ensure all __all__ exports are documented
- **Examples:** Place in examples/ directory, make them runnable
- **References:** Include paper/blog links for novel techniques
- **Versioning:** Update version in CHANGELOG when documenting releases

## Reference Files

- **README format:** `packages/llmkit/README.md`
- **CHANGELOG format:** Standard Keep a Changelog
- **Example code:** `examples/` directory (create if needed)
