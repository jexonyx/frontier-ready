#!/bin/bash
# Step 2: Organize validated backup into workspace structure
# Run this AFTER workspace restructure is complete and backup is validated

set -e

BACKUP_DIR="${HOME}/frontier-ready-vm-backup"
WORKSPACE="/Users/jex/frontier-ready"

# EDIT THESE VARIABLES:
VM_RUN_NAME="${1:-default}"        # What it was called on VM
LOCAL_EXPERIMENT="${2:-baseline}"  # What to call it in new structure

echo "================================================"
echo "STEP 2: Organizing data into workspace"
echo "================================================"
echo ""
echo "Source: ${BACKUP_DIR}/log/${VM_RUN_NAME}/"
echo "Target: ${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/"
echo ""

# Verify source exists
if [ ! -d "${BACKUP_DIR}/log/${VM_RUN_NAME}" ]; then
  echo "❌ ERROR: Source directory not found!"
  echo ""
  echo "Available run names in backup:"
  ls -d "${BACKUP_DIR}"/log/*/ 2>/dev/null || echo "  (none found)"
  echo ""
  echo "Usage: $0 [vm-run-name] [local-experiment-name]"
  echo "Example: $0 default baseline"
  echo "Example: $0 baseline baseline"
  exit 1
fi

# Verify target directory exists
if [ ! -d "${WORKSPACE}/outputs" ]; then
  echo "❌ ERROR: Workspace outputs/ directory not found!"
  echo ""
  echo "Have you completed the workspace restructure?"
  echo "Expected to find: ${WORKSPACE}/outputs/"
  echo ""
  exit 1
fi

# Create experiment output directory
echo "Creating output directory..."
mkdir -p "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}"

# Copy log file
echo ""
echo "Copying log.txt..."
if [ -f "${BACKUP_DIR}/log/${VM_RUN_NAME}/log.txt" ]; then
  cp -v "${BACKUP_DIR}/log/${VM_RUN_NAME}/log.txt" \
       "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/"
else
  echo "  ⚠️  log.txt not found"
fi

# Copy metrics
echo ""
echo "Copying metrics.jsonl..."
if [ -f "${BACKUP_DIR}/log/${VM_RUN_NAME}/metrics.jsonl" ]; then
  cp -v "${BACKUP_DIR}/log/${VM_RUN_NAME}/metrics.jsonl" \
       "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/"
else
  echo "  ⚠️  metrics.jsonl not found"
fi

# Organize checkpoints into subdirectory
echo ""
if ls "${BACKUP_DIR}/log/${VM_RUN_NAME}"/model_*.pt 1> /dev/null 2>&1; then
  echo "Organizing checkpoints..."
  mkdir -p "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/checkpoints"
  cp -v "${BACKUP_DIR}/log/${VM_RUN_NAME}"/model_*.pt \
       "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/checkpoints/"
else
  echo "  ℹ️  No checkpoints found (model_*.pt)"
fi

echo ""
echo "================================================"
echo "✅ DATA ORGANIZED"
echo "================================================"
echo ""
echo "New location: ${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/"
echo ""
echo "Contents:"
ls -lah "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/"

if [ -d "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/checkpoints" ]; then
  echo ""
  echo "Checkpoints:"
  ls -lh "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/checkpoints/" | head -10
fi

echo ""
echo "================================================"
echo "Next Steps"
echo "================================================"
echo ""
echo "1. Review data:"
echo "   cd ${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}"
echo "   cat log.txt | tail -20"
echo ""
echo "2. Run analysis (after restructure complete):"
echo "   cd ${WORKSPACE}/experiments/01-baseline"
echo "   uv run exptools-visualize --log-dir ../../outputs/${LOCAL_EXPERIMENT}"
echo ""
echo "3. Commit to git:"
echo "   git add outputs/${LOCAL_EXPERIMENT}"
echo "   git commit -m 'Add ${LOCAL_EXPERIMENT} training outputs'"
echo ""
