# Infrastructure Integration in Workspace Monorepo

## Current Infrastructure Setup

```
frontier-ready/
└── infra/                      # Pulumi + GCP
    ├── index.ts                # Main Pulumi program
    ├── resources/              # VM, disk, network resources
    ├── config/                 # Configuration loading
    ├── scripts/                # Helper scripts
    ├── Pulumi.yaml             # Pulumi project config
    ├── Pulumi.dev.yaml         # Stack configuration
    └── connect.sh              # SSH connection helper
```

**Purpose:** Provisions GPU VMs on GCP for running ML experiments

**Key Features:**
- Configurable via `experimentName` parameter
- Supports different machine types (A100, H100, L4)
- Can provision different resources per experiment
- TypeScript/Node.js based (separate from Python workspace)

---

## Integration Strategy: Hybrid Approach

### Keep Shared Infrastructure at Root

```
frontier-ready/
├── infra/                      # ✅ Stays at root (shared across all experiments)
│   ├── index.ts
│   ├── resources/
│   ├── config/
│   ├── stacks/                 # ✅ NEW: Per-experiment stack configs
│   │   ├── baseline.yaml       # Stack config for baseline experiment
│   │   ├── rope.yaml           # Stack config for rope experiment
│   │   └── README.md
│   ├── Pulumi.yaml
│   └── README.md               # Updated with workspace integration
│
├── packages/
│   └── ...
│
├── experiments/
│   ├── 01-baseline/
│   │   ├── infra.yaml          # ✅ NEW: Declares infrastructure needs
│   │   ├── deploy.sh           # ✅ NEW: Deployment helper
│   │   ├── run.py
│   │   └── ...
│   │
│   └── 02-rope/
│       ├── infra.yaml          # Different resource requirements
│       ├── deploy.sh
│       └── ...
│
└── ...
```

**Rationale:**
1. **Shared infra code** - Don't duplicate Pulumi programs
2. **Experiment-specific configs** - Each experiment declares what it needs
3. **Separation of concerns** - Infrastructure (TypeScript) vs Training (Python)
4. **Scalability** - Easy to add new experiments with different resource needs

---

## File Structure Detail

### Root Infrastructure (`infra/`)

**Keep existing structure, add:**

```
infra/
├── stacks/                     # Per-experiment stack configurations
│   ├── baseline.yaml           # Pulumi stack config for baseline
│   │   # config:
│   │   #   experimentName: baseline
│   │   #   machineType: a2-highgpu-1g
│   │   #   gpuCount: 1
│   │   #   bootDiskSize: 200
│   │
│   ├── rope.yaml               # Stack config for rope
│   ├── diffattn.yaml           # Stack config for differential attention
│   └── README.md               # How to use stacks
│
├── templates/                  # ✅ NEW: Stack templates
│   ├── single-gpu.yaml         # Template for 1-GPU experiments
│   ├── multi-gpu.yaml          # Template for distributed training
│   └── cheap.yaml              # Template for cost-optimized (L4)
│
├── scripts/
│   ├── deploy-experiment.sh   # ✅ NEW: Deploy specific experiment
│   ├── destroy-experiment.sh  # ✅ NEW: Destroy specific experiment
│   └── list-stacks.sh         # ✅ NEW: List all experiment stacks
│
└── (existing files...)
```

### Experiment-Specific Configuration

**Each experiment directory gets:**

#### `experiments/01-baseline/infra.yaml`
```yaml
# Infrastructure requirements for baseline experiment
experiment:
  name: baseline
  description: Standard GPT-2 124M training

resources:
  machine_type: a2-highgpu-1g  # 1x A100 40GB
  gpu_type: nvidia-tesla-a100
  gpu_count: 1
  boot_disk_size: 200
  data_disk_size: 500  # For datasets and checkpoints
  preemptible: true

runtime:
  zone: us-central1-a
  image_family: pytorch-latest-gpu

estimated_cost:
  hourly: "$1.35"
  total_training: "$20"  # ~15 hours at preemptible rate
```

#### `experiments/01-baseline/deploy.sh`
```bash
#!/bin/bash
# Deploy infrastructure for baseline experiment

set -e

EXPERIMENT_NAME="baseline"
INFRA_DIR="../../infra"

echo "Deploying infrastructure for experiment: ${EXPERIMENT_NAME}"

# Change to infra directory
cd "${INFRA_DIR}"

# Select or create stack
pulumi stack select "${EXPERIMENT_NAME}" 2>/dev/null || pulumi stack init "${EXPERIMENT_NAME}"

# Set configuration from experiment's infra.yaml
pulumi config set --stack "${EXPERIMENT_NAME}" experimentName "${EXPERIMENT_NAME}"
pulumi config set --stack "${EXPERIMENT_NAME}" machineType a2-highgpu-1g
pulumi config set --stack "${EXPERIMENT_NAME}" gpuType nvidia-tesla-a100
pulumi config set --stack "${EXPERIMENT_NAME}" gpuCount 1
pulumi config set --stack "${EXPERIMENT_NAME}" bootDiskSize 200
pulumi config set --stack "${EXPERIMENT_NAME}" preemptible true

# Deploy
pulumi up --stack "${EXPERIMENT_NAME}"

# Save outputs
pulumi stack output --stack "${EXPERIMENT_NAME}" --json > "../experiments/01-${EXPERIMENT_NAME}/infra-outputs.json"

echo "✅ Infrastructure deployed!"
echo "SSH command: $(pulumi stack output --stack ${EXPERIMENT_NAME} sshCommand)"
```

#### `experiments/02-rope/infra.yaml`
```yaml
# Infrastructure for RoPE variant
# Maybe needs different resources?

experiment:
  name: rope
  description: RoPE variant training

resources:
  machine_type: a2-highgpu-1g  # Same as baseline for fair comparison
  gpu_type: nvidia-tesla-a100
  gpu_count: 1
  boot_disk_size: 200
  data_disk_size: 500
  preemptible: true

runtime:
  zone: us-central1-a
  image_family: pytorch-latest-gpu
```

---

## Workflow Integration

### 1. **Deploy Infrastructure for Experiment**

```bash
cd experiments/01-baseline

# Deploy VM
./deploy.sh

# Wait for VM to be ready...
# Outputs are saved to infra-outputs.json
```

### 2. **SSH into VM**

```bash
# From experiment directory
cd experiments/01-baseline

# Use saved outputs
VM_NAME=$(jq -r '.instanceName' infra-outputs.json)
ZONE=$(jq -r '.instanceZone' infra-outputs.json)

gcloud compute ssh "${VM_NAME}" --zone="${ZONE}"
```

Or use the central connect script:
```bash
cd ../../infra
./connect.sh --experiment baseline
```

### 3. **Run Training on VM**

```bash
# On the VM
cd /data
git clone --recurse-submodules https://github.com/your-username/frontier-ready.git
cd frontier-ready

# Install packages
uv sync

# Run experiment
cd experiments/01-baseline
uv run python run.py
```

### 4. **Destroy Infrastructure**

```bash
cd experiments/01-baseline
./destroy.sh

# Or from infra directory
cd ../../infra
pulumi destroy --stack baseline
```

---

## Enhanced Infra Scripts

### `infra/scripts/deploy-experiment.sh`

```bash
#!/bin/bash
# Deploy infrastructure for any experiment
# Usage: ./deploy-experiment.sh <experiment-name>

EXPERIMENT_NAME=$1
EXPERIMENT_DIR="../experiments/*-${EXPERIMENT_NAME}"

if [ -z "$EXPERIMENT_NAME" ]; then
  echo "Usage: $0 <experiment-name>"
  echo "Example: $0 baseline"
  exit 1
fi

# Find experiment directory
EXPERIMENT_PATH=$(ls -d ../experiments/*-${EXPERIMENT_NAME} 2>/dev/null | head -1)

if [ ! -d "$EXPERIMENT_PATH" ]; then
  echo "Error: Experiment directory not found for: ${EXPERIMENT_NAME}"
  exit 1
fi

# Check for infra.yaml
if [ ! -f "$EXPERIMENT_PATH/infra.yaml" ]; then
  echo "Error: No infra.yaml found in ${EXPERIMENT_PATH}"
  exit 1
fi

echo "Deploying infrastructure for: ${EXPERIMENT_NAME}"
echo "Config: ${EXPERIMENT_PATH}/infra.yaml"

# Parse infra.yaml and set Pulumi config
# (You'd use yq or python to parse YAML)

# Select/create stack
pulumi stack select "${EXPERIMENT_NAME}" 2>/dev/null || pulumi stack init "${EXPERIMENT_NAME}"

# Deploy
pulumi up --stack "${EXPERIMENT_NAME}" --yes

# Save outputs
pulumi stack output --stack "${EXPERIMENT_NAME}" --json > "${EXPERIMENT_PATH}/infra-outputs.json"

echo "✅ Deployment complete!"
```

### `infra/scripts/list-experiments.sh`

```bash
#!/bin/bash
# List all experiment stacks

echo "Active Experiment Stacks:"
echo "========================="

pulumi stack ls --json | jq -r '.[] | "\(.name) - \(.updateInProgress // "idle")"'

echo ""
echo "To connect to an experiment VM:"
echo "  ./connect.sh --experiment <name>"
echo ""
echo "To destroy an experiment VM:"
echo "  pulumi destroy --stack <name>"
```

---

## Updated Workspace Structure

```
frontier-ready/
├── infra/                      # Shared infrastructure (TypeScript/Pulumi)
│   ├── index.ts
│   ├── resources/
│   ├── config/
│   ├── stacks/                 # Per-experiment Pulumi stack configs
│   ├── templates/              # Stack config templates
│   ├── scripts/
│   │   ├── deploy-experiment.sh
│   │   ├── destroy-experiment.sh
│   │   ├── list-experiments.sh
│   │   └── sync-outputs.sh
│   ├── package.json
│   ├── Pulumi.yaml
│   └── README.md
│
├── packages/                   # Python packages
│   ├── nanogpt/
│   └── exptools/
│
├── experiments/                # Experiment definitions
│   ├── 01-baseline/
│   │   ├── infra.yaml          # Infrastructure requirements
│   │   ├── infra-outputs.json  # Pulumi outputs (gitignored)
│   │   ├── deploy.sh           # Deploy this experiment's VM
│   │   ├── destroy.sh          # Destroy this experiment's VM
│   │   ├── run.py              # Training script
│   │   ├── config.yaml         # Training configuration
│   │   └── ...
│   │
│   └── 02-rope/
│       ├── infra.yaml
│       ├── deploy.sh
│       └── ...
│
├── data/                       # Shared datasets (on local machine)
├── outputs/                    # Training outputs (synced from VM)
└── pyproject.toml              # Workspace root
```

---

## Infrastructure Configuration Patterns

### Pattern 1: Template-Based

```bash
# experiments/01-baseline/infra.yaml
template: single-gpu-a100  # Reference to infra/templates/single-gpu-a100.yaml

overrides:
  boot_disk_size: 200      # Override specific values
  preemptible: true
```

### Pattern 2: Full Specification

```yaml
# experiments/03-large-scale/infra.yaml
resources:
  machine_type: a2-ultragpu-8g  # 8x A100
  gpu_type: nvidia-tesla-a100
  gpu_count: 8
  boot_disk_size: 500
  data_disk_size: 2000
  preemptible: false  # Can't afford interruption

runtime:
  zone: us-central1-c  # Better availability for large instances
  network: custom-vpc
```

---

## Data Synchronization Strategy

### Option 1: Shared Data Disk

Pulumi creates a persistent data disk attached to all experiment VMs:

```yaml
# infra.yaml
data:
  persistent_disk: true
  disk_name: shared-ml-data  # Shared across experiments
  mount_point: /mnt/data
```

### Option 2: Per-Experiment Data

Each experiment gets its own data disk:

```yaml
# infra.yaml
data:
  persistent_disk: true
  disk_name: "${experiment_name}-data"
  mount_point: /mnt/data
```

### Option 3: Cloud Storage

Use GCS bucket for data:

```yaml
# infra.yaml
data:
  gcs_bucket: frontier-ready-ml-data
  local_cache: /mnt/cache
```

---

## Migration Updates

Add to `MIGRATION_CHECKLIST.md`:

### Phase 5.5: Infrastructure Integration (30 min)

- [ ] Update infra/README.md with workspace integration
- [ ] Create `infra/stacks/` directory
- [ ] Create `infra/templates/` directory
- [ ] Add experiment deployment scripts:
  ```bash
  # In infra/scripts/
  ./deploy-experiment.sh
  ./destroy-experiment.sh
  ./list-experiments.sh
  ```
- [ ] For each experiment, create:
  ```bash
  # In experiments/01-baseline/
  touch infra.yaml
  touch deploy.sh
  touch destroy.sh
  chmod +x deploy.sh destroy.sh
  ```
- [ ] Add to .gitignore:
  ```
  experiments/*/infra-outputs.json
  ```
- [ ] Test deployment:
  ```bash
  cd experiments/01-baseline
  ./deploy.sh
  # Verify VM created
  ./destroy.sh
  ```

---

## Alternative: Infrastructure as Package

If you want tighter integration, could create `packages/infra-tools/`:

```
packages/
└── infra-tools/               # Python package for infra management
    ├── infratools/
    │   ├── __init__.py
    │   ├── deploy.py          # Python wrapper for Pulumi
    │   ├── config.py          # Parse infra.yaml
    │   └── sync.py            # Sync data/outputs with VM
    ├── scripts/
    │   └── infra.py           # CLI: infra-tools deploy baseline
    └── pyproject.toml

experiments/01-baseline/
└── run.py
    # Can import:
    # from infratools import deploy_vm, sync_data
```

But this adds complexity - probably overkill unless you need programmatic control.

---

## Recommendations

### For Your Current Setup

**Keep it simple:**

1. ✅ Keep `infra/` at root (shared infrastructure code)
2. ✅ Add `experiments/*/infra.yaml` (declare resource needs)
3. ✅ Add `experiments/*/deploy.sh` (convenience wrappers)
4. ✅ Add `infra/scripts/` helpers (deploy-experiment.sh, etc.)
5. ✅ Use Pulumi stacks to manage multiple experiments

**Don't:**
- ❌ Duplicate Pulumi code per experiment
- ❌ Move infra into packages/ (it's not a Python package)
- ❌ Create complex abstractions unless needed

### Benefits

- **Separation of concerns**: TypeScript infra vs Python training
- **Shared code**: One Pulumi program for all experiments
- **Flexibility**: Each experiment can request different resources
- **Simplicity**: Deploy with `./deploy.sh` from experiment dir
- **Scalability**: Easy to add experiments with different resource needs

---

## Updated Quick Start

```bash
# 1. Clone repo
git clone https://github.com/your-username/frontier-ready.git
cd frontier-ready

# 2. Set up infrastructure
cd infra
npm install
pulumi login

# 3. Deploy VM for baseline experiment
cd ../experiments/01-baseline
./deploy.sh

# 4. SSH into VM
../../infra/connect.sh --experiment baseline

# 5. On the VM: Run training
cd /data
git clone <repo>
cd frontier-ready
uv sync
cd experiments/01-baseline
uv run python run.py

# 6. (Later) Destroy VM
./destroy.sh
```

Perfect integration with the workspace structure! 🎯
