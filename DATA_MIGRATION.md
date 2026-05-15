# Data Migration: VM Outputs → Local Workspace

**IMPORTANT:** This process is **completely independent** of the workspace restructure. Handle data migration separately, on your own timeline.

---

## Overview

This guide pulls training outputs from your GCP VM and organizes them into the restructured workspace.

**Prerequisites:**
- Training has completed on VM (or you want to snapshot current state)
- VM is accessible via SSH
- Workspace restructure is complete (or at least `outputs/` directory exists)

---

## Step-by-Step Process

### Step 1: Create Redundant Backup (REQUIRED)

**Pull EVERYTHING from VM to safe local backup location.**

```bash
#!/bin/bash
# Run this first - creates full backup outside repo

VM_IP="YOUR_VM_IP_HERE"  # Replace with actual VM IP
BACKUP_DIR="${HOME}/frontier-ready-vm-backup"

echo "================================================"
echo "STEP 1: Creating redundant backup"
echo "================================================"
echo ""
echo "VM: ${VM_IP}:/data/frontier-ready/study1-gpt/build-nanogpt/log/"
echo "Backup: ${BACKUP_DIR}/log/"
echo ""

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Pull ENTIRE log directory from VM
echo "Pulling all outputs from VM..."
rsync -avz --progress \
  "${VM_IP}:/data/frontier-ready/study1-gpt/build-nanogpt/log/" \
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
ls -d "${BACKUP_DIR}"/log/*/ 2>/dev/null
echo ""
echo "================================================"
echo "⚠️  STOP HERE - MANUAL VALIDATION REQUIRED"
echo "================================================"
echo ""
echo "Before proceeding:"
echo "  1. Verify backup directory exists and has data"
echo "  2. Check log files are present and readable"
echo "  3. Verify checkpoint files exist (model_*.pt)"
echo "  4. Confirm total size is reasonable (~2-5 GB)"
echo ""
echo "To validate:"
echo "  cd ${BACKUP_DIR}/log/"
echo "  ls -lah"
echo "  cat default/log.txt | tail -20  # or check your run name"
echo "  ls default/*.pt  # check checkpoints exist"
echo ""
echo "Once validated, proceed to Step 2"
echo ""
```

**Save as:** `scripts/01-backup-vm-data.sh`

```bash
chmod +x scripts/01-backup-vm-data.sh
```

**Run it:**
```bash
./scripts/01-backup-vm-data.sh
```

---

### ⚠️ VALIDATION CHECKPOINT ⚠️

**DO NOT PROCEED until you have manually verified:**

1. **Backup exists**
   ```bash
   ls -lah ~/frontier-ready-vm-backup/log/
   ```

2. **Check run directories**
   ```bash
   ls -d ~/frontier-ready-vm-backup/log/*/
   # Should show: default/ or baseline/ or whatever run name you used
   ```

3. **Verify log file**
   ```bash
   cat ~/frontier-ready-vm-backup/log/default/log.txt | tail -20
   # Should show training logs
   ```

4. **Verify metrics**
   ```bash
   wc -l ~/frontier-ready-vm-backup/log/default/metrics.jsonl
   # Should show number of log entries
   ```

5. **Verify checkpoints**
   ```bash
   ls -lh ~/frontier-ready-vm-backup/log/default/model_*.pt
   # Should show checkpoint files (~500MB each)
   ```

6. **Check total size**
   ```bash
   du -sh ~/frontier-ready-vm-backup/log/
   # Should be 2-5 GB typically
   ```

**Only proceed once you're confident the backup is complete and valid.**

---

### Step 2: Organize Into Workspace Structure

**After workspace restructure is complete and backup is validated:**

```bash
#!/bin/bash
# Run this after restructure is complete and backup is validated

BACKUP_DIR="${HOME}/frontier-ready-vm-backup"
WORKSPACE="/Users/jex/frontier-ready"

# What run name to use from backup
VM_RUN_NAME="default"  # Change to "baseline" or whatever was used

# What experiment name in new structure
LOCAL_EXPERIMENT="baseline"  # Usually "baseline"

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
  echo "Available run names:"
  ls -d "${BACKUP_DIR}"/log/*/ 2>/dev/null
  exit 1
fi

# Verify target directory exists
if [ ! -d "${WORKSPACE}/outputs" ]; then
  echo "❌ ERROR: Workspace outputs/ directory not found!"
  echo "Have you completed the workspace restructure?"
  exit 1
fi

# Create experiment output directory
mkdir -p "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}"

# Copy data from backup to workspace
echo "Copying data..."
cp -v "${BACKUP_DIR}/log/${VM_RUN_NAME}"/log.txt \
     "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/"

cp -v "${BACKUP_DIR}/log/${VM_RUN_NAME}"/metrics.jsonl \
     "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/"

# Organize checkpoints into subdirectory
if ls "${BACKUP_DIR}/log/${VM_RUN_NAME}"/model_*.pt 1> /dev/null 2>&1; then
  echo "Organizing checkpoints..."
  mkdir -p "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/checkpoints"
  cp -v "${BACKUP_DIR}/log/${VM_RUN_NAME}"/model_*.pt \
       "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/checkpoints/"
fi

echo ""
echo "================================================"
echo "✅ DATA ORGANIZED"
echo "================================================"
echo ""
echo "New location: ${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/"
echo ""
ls -lah "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/"
echo ""
if [ -d "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/checkpoints" ]; then
  echo "Checkpoints:"
  ls -lh "${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}/checkpoints/"
  echo ""
fi

echo "Next steps:"
echo "  cd ${WORKSPACE}/outputs/${LOCAL_EXPERIMENT}"
echo "  cat log.txt | tail -20  # Review final training logs"
echo "  # Run analysis from experiments/01-baseline/"
```

**Save as:** `scripts/02-organize-data.sh`

```bash
chmod +x scripts/02-organize-data.sh
```

**Edit the script to set:**
- `VM_RUN_NAME` (what it was called on VM, probably "default")
- `LOCAL_EXPERIMENT` (what to call it locally, probably "baseline")

**Run it:**
```bash
./scripts/02-organize-data.sh
```

---

### Step 3: Commit to Git

```bash
cd /Users/jex/frontier-ready

# Review what's being added
git status
ls -lah outputs/baseline/

# Add outputs
git add outputs/baseline/

# Commit
git commit -m "Add baseline training outputs

- Training run from VM
- ~19K steps, 10B tokens
- Includes checkpoints and metrics
"
```

---

## Timeline

**Recommended order:**

1. **While training runs:** Complete workspace restructure
2. **When training completes:** Run Step 1 (backup) and validate
3. **After validation:** Run Step 2 (organize into workspace)
4. **Finally:** Commit to git

**This data migration is completely independent of the restructure.**

---

## Safety Notes

- ✅ Step 1 creates backup **outside** your repo (in `~/`)
- ✅ Backup remains even if restructure fails
- ✅ VM data is untouched (you manage VM lifecycle separately)
- ✅ Manual validation checkpoint before proceeding
- ✅ Can re-run Step 2 if needed (just copies from backup)

---

## Troubleshooting

### Can't connect to VM

```bash
# Check VM status
cd infra
pulumi stack output --stack baseline

# Get correct SSH command
./connect.sh --experiment baseline
```

### Wrong run name

```bash
# After Step 1, check what run names exist
ls -d ~/frontier-ready-vm-backup/log/*/

# Use the actual directory name in Step 2
```

### Workspace not restructured yet

That's fine! Run Step 1 now to backup data safely. Run Step 2 later after restructure is complete.

---

## Quick Reference

```bash
# Step 1: Backup from VM (run immediately)
./scripts/01-backup-vm-data.sh

# Validate backup
ls -lah ~/frontier-ready-vm-backup/log/
cat ~/frontier-ready-vm-backup/log/default/log.txt | tail

# Step 2: Organize into workspace (run after restructure)
./scripts/02-organize-data.sh

# Commit
git add outputs/baseline
git commit -m "Add baseline training outputs"
```
