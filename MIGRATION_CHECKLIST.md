# Workspace Migration Checklist

Quick reference for executing the migration to workspace monorepo structure.

---

## Pre-Migration (30 min)

- [ ] Review `RESTRUCTURE_EXECUTION_PLAN.md` thoroughly
- [ ] Review `CODE_REFACTORING_GUIDE.md` for code reorganization details
- [ ] Check current working state
  ```bash
  cd /Users/jex/frontier-ready
  git status
  ```
- [ ] Commit any uncommitted changes
  ```bash
  git add .
  git commit -m "Pre-migration: save current state"
  ```
- [ ] Create backup tag
  ```bash
  git tag -a pre-workspace-migration -m "Backup before workspace migration"
  git push origin pre-workspace-migration
  ```
- [ ] Create migration branch
  ```bash
  git checkout -b migrate-to-workspace
  ```
- [ ] Document current state
  ```bash
  tree -L 3 -I '.venv|__pycache__|.git' > MIGRATION_BEFORE.txt
  git ls-files > MIGRATION_FILES_BEFORE.txt
  ```
- [ ] Optional: Create local backup
  ```bash
  cp -r /Users/jex/frontier-ready /Users/jex/frontier-ready-backup-$(date +%Y%m%d)
  ```

---

## Phase 1: Create Directory Structure (30 min)

- [ ] Create package directories
  ```bash
  cd /Users/jex/frontier-ready
  mkdir -p packages/nanogpt/nanogpt/variants
  mkdir -p packages/nanogpt/scripts
  mkdir -p packages/exptools/exptools/scripts
  ```

- [ ] Create experiment directories
  ```bash
  mkdir -p experiments/01-baseline/writeup/figures
  mkdir -p experiments/02-rope/writeup/figures
  ```

- [ ] Create shared directories
  ```bash
  mkdir -p data/edu_fineweb10B
  mkdir -p data/hellaswag
  mkdir -p outputs/baseline
  mkdir -p outputs/rope
  ```

- [ ] Create .gitkeep files
  ```bash
  touch data/.gitkeep
  touch outputs/baseline/.gitkeep
  touch outputs/rope/.gitkeep
  ```

- [ ] Create workspace root pyproject.toml
  ```bash
  # Copy from RESTRUCTURE_EXECUTION_PLAN.md Phase 2.2
  ```

- [ ] Update .gitignore
  ```bash
  # Copy additions from RESTRUCTURE_EXECUTION_PLAN.md Phase 2.3
  ```

- [ ] Commit structure
  ```bash
  git add .
  git commit -m "Create workspace directory structure"
  ```

---

## Phase 2: Reorganize Code into nanogpt Package (1-2 hours)

**Simplified:** No careful extraction needed - just copy and reorganize!

### 2.1 Copy Source Files (5 min)
- [ ] Create temp directory and copy source code
  ```bash
  mkdir -p temp/nanogpt-source
  cp study1-gpt/build-nanogpt/train_gpt2.py temp/nanogpt-source/
  cp study1-gpt/build-nanogpt/fineweb.py temp/nanogpt-source/
  cp study1-gpt/build-nanogpt/hellaswag.py temp/nanogpt-source/
  cp study1-gpt/build-nanogpt/variant_rope.py temp/nanogpt-source/
  ```

### 2.2 Create Package Structure (5 min)
- [ ] Create directory structure
  ```bash
  mkdir -p packages/nanogpt/nanogpt/variants
  mkdir -p packages/nanogpt/scripts
  ```

### 2.3 Reorganize into Modules (1-1.5 hours)

**Note:** Just copy classes/functions and reorganize - don't worry about preserving structure!

- [ ] Create `packages/nanogpt/nanogpt/__init__.py`
  - Set up package exports

- [ ] Create `packages/nanogpt/nanogpt/config.py`
  - Copy `GPTConfig` dataclass from train_gpt2.py
  - Add `TrainingConfig` class

- [ ] Create `packages/nanogpt/nanogpt/model.py`
  - Copy model classes: CausalSelfAttention, MLP, Block, GPT
  - Reorganize as makes sense

- [ ] Create `packages/nanogpt/nanogpt/training.py`
  - Copy training loop code
  - Create `Trainer` class to encapsulate it
  - Reorganize methods logically

- [ ] Create `packages/nanogpt/nanogpt/data.py`
  - Copy `DataLoaderLite` from train_gpt2.py
  - Copy data prep logic from fineweb.py

- [ ] Create `packages/nanogpt/nanogpt/eval.py`
  - Copy HellaSwag evaluation from hellaswag.py

- [ ] Create `packages/nanogpt/nanogpt/variants/rope.py`
  - Copy RoPE variant from variant_rope.py

- [ ] Create CLI scripts in `packages/nanogpt/scripts/`
  - train.py, prepare_data.py, evaluate.py

### 2.6 CLI Scripts (30 min)
- [ ] Create `packages/nanogpt/scripts/train.py`
  - CLI wrapper for Trainer
  - Argument parsing
  - Entry point
- [ ] Make executable
  ```bash
  chmod +x packages/nanogpt/scripts/train.py
  ```

### 2.7 Package Configuration (15 min)
- [ ] Create `packages/nanogpt/pyproject.toml`
  - Copy from CODE_REFACTORING_GUIDE.md
  - Set up entry points for scripts
- [ ] Create `packages/nanogpt/README.md`

### 2.8 Install and Test (30 min)
- [ ] Install package in development mode
  ```bash
  cd /Users/jex/frontier-ready
  uv pip install -e packages/nanogpt
  ```
- [ ] Test imports
  ```bash
  python -c "from nanogpt import GPT, Trainer; print('✓ nanogpt package works')"
  ```
- [ ] Test CLI
  ```bash
  nanogpt-train --help
  ```

- [ ] Commit nanogpt package
  ```bash
  git add packages/nanogpt
  git commit -m "Add nanogpt package"
  ```

### 2.9 Add Attribution (10 min)
- [ ] Add acknowledgment to packages/nanogpt/README.md
  ```markdown
  ## Acknowledgments

  This package is built on [Andrej Karpathy's build-nanogpt](https://github.com/karpathy/build-nanogpt),
  which provides a minimal, educational implementation of GPT-2. We've restructured and extended
  this codebase for architectural experimentation.

  **Original implementation:** Copyright (c) Andrej Karpathy
  **License:** MIT (original)
  ```

- [ ] Also add to root README.md (if not already present)

- [ ] Commit acknowledgment
  ```bash
  git add packages/nanogpt/README.md README.md
  git commit -m "Add acknowledgment of Karpathy's build-nanogpt"
  ```

---

## Phase 3: Create exptools Package (1 hour)

### 3.1 Package Structure (15 min)
- [ ] Create `packages/exptools/exptools/__init__.py`
- [ ] Copy `study1-gpt/analysis/parse_metrics.py` → `packages/exptools/exptools/metrics.py`
- [ ] Update imports in metrics.py

### 3.2 Visualization Module (15 min)
- [ ] Copy `study1-gpt/analysis/visualize_run.py` → `packages/exptools/exptools/visualization.py`
- [ ] Create `packages/exptools/scripts/visualize_run.py` (CLI wrapper)
- [ ] Update imports

### 3.3 Summary Module (15 min)
- [ ] Copy `study1-gpt/analysis/summarize_run.py` → `packages/exptools/exptools/summary.py`
- [ ] Create `packages/exptools/scripts/summarize_run.py` (CLI wrapper)
- [ ] Update imports

### 3.4 Package Configuration (15 min)
- [ ] Create `packages/exptools/pyproject.toml`
- [ ] Create `packages/exptools/README.md`
- [ ] Install package
  ```bash
  uv pip install -e packages/exptools
  ```
- [ ] Test imports
  ```bash
  python -c "from exptools import load_metrics; print('✓ exptools package works')"
  ```

- [ ] Commit exptools package
  ```bash
  git add packages/exptools
  git commit -m "Add exptools package"
  ```

---

## Phase 4: Create Experiments (1 hour)

### 4.1 Baseline Experiment (30 min)
- [ ] Create `experiments/01-baseline/run.py`
  - Import from nanogpt package
  - Set up config
  - Run training
- [ ] Create `experiments/01-baseline/config.yaml`
- [ ] Create `experiments/01-baseline/analysis.ipynb`
- [ ] Create `experiments/01-baseline/pyproject.toml`
- [ ] Create `experiments/01-baseline/README.md`
- [ ] Move existing writeup
  ```bash
  cp -r study1-gpt/writeup/* experiments/01-baseline/writeup/
  ```

### 4.2 RoPE Experiment (30 min)
- [ ] Copy baseline structure to 02-rope
  ```bash
  cp -r experiments/01-baseline/* experiments/02-rope/
  ```
- [ ] Update config.yaml for RoPE variant
- [ ] Update README.md
- [ ] Update run.py to use RoPE variant

- [ ] Commit experiments
  ```bash
  git add experiments
  git commit -m "Add experiment directories"
  ```

---

## Phase 5: Move Data and Configure Workspace (30 min)

### 5.1 Move Data
- [ ] Move FineWeb data
  ```bash
  mv study1-gpt/build-nanogpt/edu_fineweb10B/* data/edu_fineweb10B/ 2>/dev/null || true
  ```
- [ ] Move HellaSwag data
  ```bash
  mv study1-gpt/build-nanogpt/hellaswag/* data/hellaswag/ 2>/dev/null || true
  ```
- [ ] Verify data moved correctly
  ```bash
  ls -lh data/edu_fineweb10B | head
  ls -lh data/hellaswag
  ```

### 5.2 Configure Workspace
- [ ] Update workspace pyproject.toml with all members
- [ ] Create uv.lock
  ```bash
  cd /Users/jex/frontier-ready
  uv sync
  ```

- [ ] Commit data moves
  ```bash
  git add data/
  git commit -m "Move data to shared location"
  ```

---

## Phase 5.5: Infrastructure Integration (45 min)

### 5.5.1 Update Infrastructure Directory (15 min)
- [ ] Create stacks directory
  ```bash
  cd infra
  mkdir -p stacks templates
  ```

- [ ] Create stack templates
  ```bash
  cat > templates/single-gpu-a100.yaml << 'EOF'
# Template for single A100 GPU experiments
config:
  machineType: a2-highgpu-1g
  gpuType: nvidia-tesla-a100
  gpuCount: 1
  bootDiskSize: 200
  dataDiskSize: 500
  preemptible: true
  zone: us-central1-a
  imageFamily: pytorch-latest-gpu
EOF
  ```

- [ ] Create deployment helper scripts
  ```bash
  cat > scripts/deploy-experiment.sh << 'EOF'
#!/bin/bash
# Deploy infrastructure for specific experiment
# Usage: ./deploy-experiment.sh <experiment-name>

EXPERIMENT_NAME=$1

if [ -z "$EXPERIMENT_NAME" ]; then
  echo "Usage: $0 <experiment-name>"
  echo "Example: $0 baseline"
  exit 1
fi

# Find experiment directory
EXPERIMENT_PATH=$(ls -d ../experiments/*-${EXPERIMENT_NAME} 2>/dev/null | head -1)

if [ ! -d "$EXPERIMENT_PATH" ]; then
  echo "Error: Experiment not found: ${EXPERIMENT_NAME}"
  exit 1
fi

echo "Deploying infrastructure for: ${EXPERIMENT_NAME}"

# Select/create Pulumi stack
pulumi stack select "${EXPERIMENT_NAME}" 2>/dev/null || pulumi stack init "${EXPERIMENT_NAME}"

# Set config from experiment's infra.yaml (simplified version)
pulumi config set experimentName "${EXPERIMENT_NAME}"

# Deploy
pulumi up --stack "${EXPERIMENT_NAME}"

# Save outputs
pulumi stack output --stack "${EXPERIMENT_NAME}" --json > "${EXPERIMENT_PATH}/infra-outputs.json"

echo "✅ Infrastructure deployed!"
echo "SSH: $(pulumi stack output --stack ${EXPERIMENT_NAME} sshCommand)"
EOF

  chmod +x scripts/deploy-experiment.sh
  ```

- [ ] Create list helper
  ```bash
  cat > scripts/list-experiments.sh << 'EOF'
#!/bin/bash
# List all experiment stacks

echo "Active Experiment Stacks:"
echo "========================="
pulumi stack ls

echo ""
echo "To deploy: ./scripts/deploy-experiment.sh <name>"
echo "To destroy: pulumi destroy --stack <name>"
EOF

  chmod +x scripts/list-experiments.sh
  ```

- [ ] Update infra README with workspace integration
  ```bash
  # Add section about workspace integration
  # Reference INFRASTRUCTURE_INTEGRATION.md
  ```

### 5.5.2 Add Experiment Infrastructure Configs (20 min)

- [ ] Create baseline infrastructure config
  ```bash
  cd ../experiments/01-baseline

  cat > infra.yaml << 'EOF'
# Infrastructure requirements for baseline experiment
experiment:
  name: baseline
  description: Standard GPT-2 124M training

resources:
  machine_type: a2-highgpu-1g  # 1x A100 40GB
  gpu_type: nvidia-tesla-a100
  gpu_count: 1
  boot_disk_size: 200
  data_disk_size: 500
  preemptible: true

runtime:
  zone: us-central1-a
  image_family: pytorch-latest-gpu

estimated_cost:
  hourly: "$1.35"
  total_training: "$20"  # ~15 hours
EOF
  ```

- [ ] Create baseline deployment script
  ```bash
  cat > deploy.sh << 'EOF'
#!/bin/bash
# Deploy infrastructure for baseline experiment

set -e

EXPERIMENT_NAME="baseline"
INFRA_DIR="../../infra"

echo "Deploying infrastructure for experiment: ${EXPERIMENT_NAME}"

cd "${INFRA_DIR}"

# Use the shared deployment script
./scripts/deploy-experiment.sh "${EXPERIMENT_NAME}"

echo "✅ VM ready! Connect with: cd infra && ./connect.sh --experiment ${EXPERIMENT_NAME}"
EOF

  chmod +x deploy.sh
  ```

- [ ] Create baseline destroy script
  ```bash
  cat > destroy.sh << 'EOF'
#!/bin/bash
# Destroy infrastructure for baseline experiment

set -e

EXPERIMENT_NAME="baseline"
INFRA_DIR="../../infra"

echo "Destroying infrastructure for experiment: ${EXPERIMENT_NAME}"

cd "${INFRA_DIR}"

pulumi destroy --stack "${EXPERIMENT_NAME}" --yes

echo "✅ Infrastructure destroyed"
EOF

  chmod +x destroy.sh
  ```

- [ ] Create rope infrastructure config
  ```bash
  cd ../02-rope

  # Copy from baseline (same resources for fair comparison)
  cp ../01-baseline/infra.yaml .
  sed -i.bak 's/name: baseline/name: rope/' infra.yaml
  sed -i.bak 's/description: Standard GPT-2 124M/description: RoPE variant/' infra.yaml
  rm infra.yaml.bak
  ```

- [ ] Create rope deploy/destroy scripts
  ```bash
  # Copy and modify from baseline
  cp ../01-baseline/deploy.sh .
  cp ../01-baseline/destroy.sh .
  sed -i.bak 's/EXPERIMENT_NAME="baseline"/EXPERIMENT_NAME="rope"/' deploy.sh
  sed -i.bak 's/EXPERIMENT_NAME="baseline"/EXPERIMENT_NAME="rope"/' destroy.sh
  rm *.bak
  ```

### 5.5.3 Update .gitignore (5 min)

- [ ] Add infrastructure outputs to gitignore
  ```bash
  cd /Users/jex/frontier-ready

  cat >> .gitignore << 'EOF'

# Infrastructure outputs (per-experiment)
experiments/*/infra-outputs.json
experiments/*/.pulumi/

# Pulumi state (managed remotely)
infra/.pulumi/
EOF
  ```

### 5.5.4 Test Infrastructure Integration (5 min)

- [ ] Verify infra scripts are executable
  ```bash
  ls -la infra/scripts/*.sh
  ls -la experiments/01-baseline/deploy.sh
  ls -la experiments/01-baseline/destroy.sh
  ```

- [ ] Test experiment config is valid
  ```bash
  cat experiments/01-baseline/infra.yaml
  cat experiments/02-rope/infra.yaml
  ```

- [ ] Optionally: Test deployment (if you have GCP credits)
  ```bash
  cd experiments/01-baseline
  # Review what will be deployed
  cat infra.yaml
  # Uncomment to actually deploy:
  # ./deploy.sh
  # ./destroy.sh
  ```

- [ ] Commit infrastructure integration
  ```bash
  git add infra/ experiments/*/infra.yaml experiments/*/deploy.sh experiments/*/destroy.sh
  git commit -m "Add infrastructure integration for experiments"
  ```

---

## Phase 6: Testing (1-2 hours)

### 6.1 Unit Tests
- [ ] Test nanogpt model
  ```bash
  python -c "from nanogpt import GPT, GPTConfig; GPT(GPTConfig())"
  ```
- [ ] Test nanogpt training (dry run)
  ```bash
  # Create test with max_steps=1
  ```
- [ ] Test exptools
  ```bash
  # Test with existing metrics if available
  ```

### 6.2 Integration Tests
- [ ] Test experiment can import packages
  ```bash
  cd experiments/01-baseline
  python -c "import run; print('✓ Can import run.py')"
  ```
- [ ] Test workspace install
  ```bash
  cd /Users/jex/frontier-ready
  uv sync
  ```
- [ ] Test CLI tools
  ```bash
  nanogpt-train --help
  exptools-visualize --help
  exptools-summarize --help
  ```

### 6.3 Comparison Test (Optional)
- [ ] Run quick training test with old code
  ```bash
  cd study1-gpt/build-nanogpt
  python train_gpt2.py --max-steps 10
  ```
- [ ] Run quick training test with new code
  ```bash
  cd /Users/jex/frontier-ready
  nanogpt-train --max-steps 10 --output-dir outputs/test
  ```
- [ ] Compare outputs (losses should be similar)

---

## Phase 7: Cleanup (30 min)

### 7.1 Remove Old Structure
- [ ] Verify everything is migrated (check twice!)
- [ ] Remove git submodule
  ```bash
  git submodule deinit -f study1-gpt/build-nanogpt
  git rm -f study1-gpt/build-nanogpt
  rm -f .gitmodules  # If no other submodules
  ```
- [ ] Remove old study1-gpt directory
  ```bash
  git rm -rf study1-gpt
  ```

- [ ] Commit removal
  ```bash
  git commit -m "Remove old study1-gpt structure after migration"
  ```

### 7.2 Update Documentation
- [ ] Update root README.md
  - Document new structure
  - Update quick start
  - Update project layout
- [ ] Create `MIGRATION_COMPLETE.md`
- [ ] Update `PLAN.md` if needed

- [ ] Commit documentation
  ```bash
  git add README.md MIGRATION_COMPLETE.md
  git commit -m "Update documentation for workspace structure"
  ```

---

## Phase 8: Finalization (30 min)

### 8.1 Final Verification
- [ ] All packages install: `uv sync`
- [ ] All imports work
- [ ] No broken paths
- [ ] Data accessible from experiments
- [ ] Can run training (at least dry run)
- [ ] Can run analysis scripts

### 8.2 Documentation
- [ ] Verify all READMEs are accurate
- [ ] Check pyproject.toml files are correct
- [ ] Ensure .gitignore is complete

### 8.3 Git Housekeeping
- [ ] Review all commits
  ```bash
  git log --oneline migrate-to-workspace
  ```
- [ ] Squash if needed (optional)
- [ ] Write comprehensive merge commit message

### 8.4 Merge to Main
- [ ] Final check
  ```bash
  git status
  git diff main..migrate-to-workspace --stat
  ```
- [ ] Merge migration branch
  ```bash
  git checkout main
  git merge migrate-to-workspace
  ```
- [ ] Tag the migration
  ```bash
  git tag -a v0.2.0-workspace -m "Workspace monorepo structure"
  ```
- [ ] Push to remote
  ```bash
  git push origin main
  git push origin v0.2.0-workspace
  ```

---

## Post-Migration

- [ ] Test from fresh clone
  ```bash
  cd /tmp
  git clone YOUR_REPO
  cd frontier-ready
  uv sync
  python -c "from nanogpt import GPT; from exptools import load_metrics; print('✓ All good')"
  ```

- [ ] Update CI/CD if applicable
- [ ] Notify collaborators of new structure
- [ ] Remove backup if everything works
  ```bash
  rm -rf /Users/jex/frontier-ready-backup-*
  ```

---

## Troubleshooting

### Import Errors
```bash
# Reinstall packages
uv sync --reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

### Missing Data
```bash
# Verify data location
ls -lh data/edu_fineweb10B
ls -lh data/hellaswag

# Check experiment can access
cd experiments/01-baseline
python -c "from pathlib import Path; print(Path('../../data/edu_fineweb10B').exists())"
```

### Training Fails
```bash
# Check device
python -c "import torch; print(torch.cuda.is_available())"

# Run with verbose
nanogpt-train --help
nanogpt-train --max-steps 1 --output-dir outputs/debug
```

---

## Rollback (If Needed)

```bash
# Return to pre-migration state
git checkout pre-workspace-migration

# Or from backup
cp -r /Users/jex/frontier-ready-backup-YYYYMMDD/* /Users/jex/frontier-ready/
```

---

## Progress Tracking

**Started:** ___________
**Completed:** ___________
**Total Time:** ___________ hours

**Notes:**
-
-
-

---

## Success Criteria

✅ All packages install without errors
✅ All imports work
✅ Can run training (at least dry run)
✅ Can run analysis scripts
✅ Data accessible from experiments
✅ No broken paths or symlinks
✅ Documentation is accurate
✅ Git history is clean
✅ Tests pass (if any)
✅ Fresh clone works
