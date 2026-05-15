#!/bin/bash
# Step 1: Create redundant backup of ALL VM outputs
# This is completely independent of workspace restructure

set -e

VM_IP="${1}"

if [ -z "$VM_IP" ]; then
  echo "Usage: $0 <vm-ip>"
  echo ""
  echo "Example:"
  echo "  $0 34.123.45.67"
  echo ""
  exit 1
fi

BACKUP_DIR="${HOME}/frontier-ready-vm-backup"
VM_PATH="/data/frontier-ready/study1-gpt/build-nanogpt/log/"

echo "================================================"
echo "STEP 1: Creating redundant backup"
echo "================================================"
echo ""
echo "VM:     ${VM_IP}:${VM_PATH}"
echo "Backup: ${BACKUP_DIR}/log/"
echo ""

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Pull ENTIRE log directory from VM
echo "Pulling all outputs from VM..."
echo ""

rsync -avz --progress \
  "${VM_IP}:${VM_PATH}" \
  "${BACKUP_DIR}/log/"

echo ""
echo "================================================"
echo "✅ BACKUP COMPLETE"
echo "================================================"
echo ""
echo "Backup location: ${BACKUP_DIR}/log/"
echo ""
echo "Contents:"
ls -lah "${BACKUP_DIR}/log/"
echo ""
echo "Run names found:"
ls -d "${BACKUP_DIR}"/log/*/ 2>/dev/null || echo "  (none - check if path is correct)"
echo ""
echo "================================================"
echo "⚠️  STOP HERE - MANUAL VALIDATION REQUIRED"
echo "================================================"
echo ""
echo "Before proceeding to Step 2, validate the backup:"
echo ""
echo "1. Check backup directory exists:"
echo "   ls -lah ${BACKUP_DIR}/log/"
echo ""
echo "2. Check run directories:"
echo "   ls -d ${BACKUP_DIR}/log/*/"
echo ""
echo "3. Verify log file (adjust run name if needed):"
echo "   cat ${BACKUP_DIR}/log/default/log.txt | tail -20"
echo ""
echo "4. Verify metrics:"
echo "   wc -l ${BACKUP_DIR}/log/default/metrics.jsonl"
echo ""
echo "5. Verify checkpoints:"
echo "   ls -lh ${BACKUP_DIR}/log/default/model_*.pt"
echo ""
echo "6. Check total size:"
echo "   du -sh ${BACKUP_DIR}/log/"
echo ""
echo "Once validated, run:"
echo "  ./scripts/02-organize-data.sh"
echo ""
