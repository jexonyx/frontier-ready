# Deploy Infrastructure Skill

Sets up GCP infrastructure for experiments using Pulumi, with cost estimation and validation.

## Usage

```
/deploy-infra
```

## What This Skill Does

1. Prompts for deployment configuration (type, machine, region, disk)
2. Validates GCP setup (gcloud auth, project)
3. Checks existing infrastructure to avoid conflicts
4. Calculates costs (hourly, daily, total for experiment)
5. Creates/updates Pulumi stack configuration
6. Runs deployment preview
7. Shows cost estimate and gets user confirmation
8. Deploys infrastructure (VM, disks, networking)
9. Generates connection scripts (connect.sh, setup.sh)
10. Validates deployment (VM reachable, GPU available)
11. Stages files with git (user handles commit)

## Execution Steps

### Step 1: Gather Information

Use the `AskUserQuestion` tool to collect deployment details:

**Questions to ask:**

1. **Deployment Type** (header: "Type")
   - Question: "What type of infrastructure are you deploying?"
   - Options:
     - label: "Experiment VM", description: "GPU VM for running a specific experiment"
     - label: "Shared compute", description: "Shared GPU VM for multiple experiments"
     - label: "Storage", description: "Data storage bucket only"

2. **Experiment Name** (header: "Experiment") [If type is "Experiment VM"]
   - Question: "Which experiment is this for? (e.g., '02-rope', 'baseline')"
   - Options:
     - (Auto-detect from experiments/ directory)
     - label: "01-baseline", description: "Baseline GPT-2 124M"
     - label: "02-rope", description: "RoPE position embeddings"

3. **Machine Type** (header: "Machine")
   - Question: "What GPU machine type do you need?"
   - Options:
     - label: "a2-highgpu-1g", description: "1x A100 40GB - $1.35/hr (good for 124M models)"
     - label: "a2-ultragpu-1g", description: "1x A100 80GB - $1.60/hr (more memory)"
     - label: "a2-highgpu-2g", description: "2x A100 40GB - $2.70/hr (distributed training)"
     - label: "a2-highgpu-4g", description: "4x A100 40GB - $5.40/hr (large-scale training)"

4. **Region** (header: "Region")
   - Question: "Which GCP region?"
   - Options:
     - label: "us-central1-a", description: "Iowa (default, good availability)"
     - label: "us-west1-b", description: "Oregon (low latency west coast)"
     - label: "us-east1-b", description: "South Carolina (low latency east coast)"
     - label: "europe-west4-a", description: "Netherlands (EU region)"

5. **Data Disk Size** (header: "Disk")
   - Question: "How much data disk space do you need?"
   - Options:
     - label: "500GB", description: "$50/month (~10B tokens) (default)"
     - label: "1TB", description: "$100/month (~20B tokens)"
     - label: "2TB", description: "$200/month (~40B tokens)"

6. **Preemptible** (header: "Preemptible")
   - Question: "Use preemptible VM? (60-90% cost savings, may be interrupted)"
   - Options:
     - label: "Yes", description: "Preemptible - much cheaper but may restart (recommended for experiments)"
     - label: "No", description: "On-demand - stable but expensive"

7. **Budget Alert** (header: "Budget")
   - Question: "Set budget alert threshold?"
   - Options:
     - label: "$100", description: "Alert at $100 spend"
     - label: "$500", description: "Alert at $500 spend"
     - label: "$1000", description: "Alert at $1000 spend"
     - label: "None", description: "No budget alert"

### Step 2: Validate GCP Setup

Check GCP authentication and project:

```bash
# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found"
    echo "Install: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with GCP"
    echo "Run: gcloud auth login"
    exit 1
fi

# Get current project
GCP_PROJECT=$(gcloud config get-value project)
if [ -z "${GCP_PROJECT}" ]; then
    echo "❌ No GCP project set"
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "✓ GCP authenticated"
echo "  Project: ${GCP_PROJECT}"
echo "  Account: $(gcloud config get-value account)"
```

### Step 3: Check Existing Infrastructure

Check if infrastructure already exists for this experiment:

```bash
cd infra

# Check if Pulumi stack exists
STACK_NAME="${EXPERIMENT_NAME}"

if pulumi stack ls | grep -q "${STACK_NAME}"; then
    echo "⚠️  Stack ${STACK_NAME} already exists"

    # Show current stack info
    pulumi stack select "${STACK_NAME}"
    pulumi stack output

    # Ask user what to do
    # Options: "Update existing", "Destroy and recreate", "Cancel"
fi

# Check if VM is already running
VM_NAME="${EXPERIMENT_NAME}-vm"
if gcloud compute instances list --filter="name:${VM_NAME}" --format="value(name)" | grep -q .; then
    echo "⚠️  VM ${VM_NAME} already exists"
    echo "Status: $(gcloud compute instances describe ${VM_NAME} --zone=${ZONE} --format='value(status)')"

    # Ask if they want to continue
fi
```

### Step 4: Calculate Costs

Calculate hourly, daily, and total experiment costs:

```bash
# Machine costs (per hour)
case "${MACHINE_TYPE}" in
    "a2-highgpu-1g")
        MACHINE_COST_HR=1.35
        ;;
    "a2-ultragpu-1g")
        MACHINE_COST_HR=1.60
        ;;
    "a2-highgpu-2g")
        MACHINE_COST_HR=2.70
        ;;
    "a2-highgpu-4g")
        MACHINE_COST_HR=5.40
        ;;
esac

# Apply preemptible discount (70% savings)
if [ "${PREEMPTIBLE}" = "Yes" ]; then
    MACHINE_COST_HR=$(echo "${MACHINE_COST_HR} * 0.3" | bc)
fi

# Disk costs (per month, convert to hourly)
DISK_SIZE_GB=$(echo "${DISK_SIZE}" | sed 's/GB//' | sed 's/TB/*1024/')
DISK_COST_MONTH=$(echo "${DISK_SIZE_GB} * 0.1" | bc)  # $0.10/GB/month
DISK_COST_HR=$(echo "${DISK_COST_MONTH} / 730" | bc -l)  # 730 hours/month

# Total costs
TOTAL_COST_HR=$(echo "${MACHINE_COST_HR} + ${DISK_COST_HR}" | bc -l)
TOTAL_COST_DAY=$(echo "${TOTAL_COST_HR} * 24" | bc -l)

# Estimate experiment duration (default: 48 hours for 10B tokens)
EXPERIMENT_HOURS=48
TOTAL_COST=$(echo "${TOTAL_COST_HR} * ${EXPERIMENT_HOURS}" | bc -l)

echo "💰 Cost Estimate:"
echo "  Machine: \$${MACHINE_COST_HR}/hr (${MACHINE_TYPE}$([ \"${PREEMPTIBLE}\" = \"Yes\" ] && echo \" preemptible\" || echo \"\"))"
echo "  Disk: \$${DISK_COST_HR}/hr (${DISK_SIZE})"
echo "  Total: \$${TOTAL_COST_HR}/hr"
echo ""
echo "  Per day: \$${TOTAL_COST_DAY}/day"
echo "  Est. experiment (48hrs): \$${TOTAL_COST}"
```

### Step 5: Create Pulumi Stack Configuration

Create or update Pulumi stack YAML:

```bash
STACK_FILE="infra/Pulumi.${STACK_NAME}.yaml"
```

**Template for `Pulumi.{stack}.yaml`:**

```yaml
config:
  gcp:project: {GCP_PROJECT}
  gcp:region: {REGION}
  gcp:zone: {ZONE}
  frontier-ready-infra:experimentName: {EXPERIMENT_NAME}
  frontier-ready-infra:machineType: {MACHINE_TYPE}
  frontier-ready-infra:diskSize: {DISK_SIZE_GB}
  frontier-ready-infra:preemptible: {PREEMPTIBLE_BOOL}
  frontier-ready-infra:budgetAlert: {BUDGET_ALERT}
```

### Step 6: Preview Deployment

Run Pulumi preview to show what will be created:

```bash
cd infra

# Select stack
pulumi stack select "${STACK_NAME}"

# Run preview
echo "Running deployment preview..."
pulumi preview

# Capture resource count
RESOURCE_COUNT=$(pulumi preview --json | jq '.changeSummary.create' 2>/dev/null || echo "unknown")

echo ""
echo "📋 Deployment will create ${RESOURCE_COUNT} resources:"
echo "  - Compute instance (${MACHINE_TYPE})"
echo "  - Boot disk (200GB)"
echo "  - Data disk (${DISK_SIZE})"
echo "  - Firewall rules"
echo "  - Service account"
```

### Step 7: Show Cost Estimate and Get Confirmation

Display summary and ask for confirmation:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
           DEPLOYMENT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Experiment:  {EXPERIMENT_NAME}
Machine:     {MACHINE_TYPE} (${PREEMPTIBLE_TEXT})
Region:      {REGION}
Disk:        {DISK_SIZE}

💰 COSTS:
  Hourly:    ${TOTAL_COST_HR}/hr
  Daily:     ${TOTAL_COST_DAY}/day
  48hr exp:  ${TOTAL_COST}

⚠️  IMPORTANT:
  - Remember to destroy VM after experiment
  - Preemptible VMs may restart during training
  - Budget alert set at: {BUDGET_ALERT}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Continue with deployment? (yes/no):
```

### Step 8: Deploy Infrastructure

If confirmed, run deployment:

```bash
echo "Starting deployment..."
pulumi up --yes

# Capture outputs
VM_IP=$(pulumi stack output vmExternalIp)
VM_NAME=$(pulumi stack output vmName)
VM_ZONE=$(pulumi stack output vmZone)

echo ""
echo "✅ Deployment complete!"
echo "  VM Name: ${VM_NAME}"
echo "  External IP: ${VM_IP}"
echo "  Zone: ${VM_ZONE}"
```

### Step 9: Generate Connection Scripts

Create convenience scripts for connecting to the VM:

**Update `infra/connect.sh`:**

```bash
#!/bin/bash
# Connect to experiment VM

EXPERIMENT_NAME="${1:-baseline}"

# Get VM info from Pulumi
cd infra
pulumi stack select "${EXPERIMENT_NAME}"

VM_NAME=$(pulumi stack output vmName)
VM_ZONE=$(pulumi stack output vmZone)

if [ -z "${VM_NAME}" ]; then
    echo "❌ No VM found for experiment: ${EXPERIMENT_NAME}"
    echo "Available experiments:"
    pulumi stack ls
    exit 1
fi

echo "Connecting to ${VM_NAME} in ${VM_ZONE}..."
gcloud compute ssh "${VM_NAME}" --zone="${VM_ZONE}"
```

**Create `infra/setup-{experiment}.sh` for VM initialization:**

```bash
#!/bin/bash
# Setup script for {EXPERIMENT_NAME} VM
# Run this on the VM after first connection

set -e

echo "Setting up {EXPERIMENT_NAME} environment..."

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone repository
if [ ! -d "frontier-ready" ]; then
    git clone YOUR_REPO_URL frontier-ready
fi

cd frontier-ready

# Sync workspace
uv sync

# Download data if needed
if [ ! -d "data/edu_fineweb10B" ]; then
    echo "Downloading dataset..."
    # Add data download commands
fi

# Verify GPU
nvidia-smi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start training:"
echo "  cd experiments/{EXPERIMENT_NAME}"
echo "  uv run python run.py"
```

**Create `infra/destroy-{experiment}.sh`:**

```bash
#!/bin/bash
# Destroy infrastructure for {EXPERIMENT_NAME}

set -e

EXPERIMENT_NAME="{EXPERIMENT_NAME}"

echo "⚠️  WARNING: This will destroy all infrastructure for ${EXPERIMENT_NAME}"
echo "This includes:"
echo "  - VM instance"
echo "  - All disks (data will be lost!)"
echo "  - Firewall rules"
echo ""

read -p "Type experiment name to confirm: " CONFIRM

if [ "${CONFIRM}" != "${EXPERIMENT_NAME}" ]; then
    echo "Cancelled"
    exit 1
fi

cd infra
pulumi stack select "${EXPERIMENT_NAME}"
pulumi destroy --yes

echo ""
echo "✅ Infrastructure destroyed"
echo "Stack still exists. To remove completely: pulumi stack rm ${EXPERIMENT_NAME}"
```

### Step 10: Validate Deployment

Check that the VM is running and accessible:

```bash
echo "Validating deployment..."

# 1. Check VM status
VM_STATUS=$(gcloud compute instances describe "${VM_NAME}" --zone="${VM_ZONE}" --format='value(status)')

if [ "${VM_STATUS}" != "RUNNING" ]; then
    echo "⚠️  VM is not running (status: ${VM_STATUS})"
    echo "Waiting for VM to start..."
    sleep 30
fi

# 2. Check SSH connectivity
echo "Testing SSH connection..."
if gcloud compute ssh "${VM_NAME}" --zone="${VM_ZONE}" --command="echo 'SSH OK'" 2>/dev/null; then
    echo "✓ SSH connection successful"
else
    echo "⚠️  SSH connection failed (VM may still be booting)"
fi

# 3. Check GPU availability (if machine has GPU)
if [[ "${MACHINE_TYPE}" == a2-* ]]; then
    echo "Checking GPU..."
    GPU_COUNT=$(gcloud compute ssh "${VM_NAME}" --zone="${VM_ZONE}" --command="nvidia-smi --list-gpus | wc -l" 2>/dev/null || echo "0")

    if [ "${GPU_COUNT}" -gt 0 ]; then
        echo "✓ ${GPU_COUNT} GPU(s) detected"
    else
        echo "⚠️  No GPUs detected (drivers may still be installing)"
    fi
fi

echo ""
echo "✅ Deployment validated!"
```

### Step 11: Stage Files with Git

```bash
cd /Users/jex/frontier-ready

# Stage Pulumi stack config
git add infra/Pulumi.${EXPERIMENT_NAME}.yaml

# Stage connection scripts
git add infra/connect.sh
git add infra/setup-${EXPERIMENT_NAME}.sh
git add infra/destroy-${EXPERIMENT_NAME}.sh

git status
```

### Step 12: Summary Output

```
✅ Infrastructure deployed successfully!

🖥️  VM Details:
  Name:        {VM_NAME}
  Type:        {MACHINE_TYPE}
  Zone:        {VM_ZONE}
  External IP: {VM_IP}
  Status:      RUNNING
  GPUs:        {GPU_COUNT}x A100

💾 Storage:
  Boot disk:   200GB
  Data disk:   {DISK_SIZE}
  Total:       {TOTAL_DISK}

💰 Costs:
  Hourly:      ${TOTAL_COST_HR}/hr
  Daily:       ${TOTAL_COST_DAY}/day
  Budget alert: {BUDGET_ALERT}

📝 Suggested commit message:

Deploy infrastructure for {EXPERIMENT_NAME}

- Machine: {MACHINE_TYPE} ({PREEMPTIBLE_TEXT})
- Region: {REGION}
- Disk: {DISK_SIZE}
- Cost: ${TOTAL_COST_HR}/hr

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

📦 Next steps:
  1. Connect to VM: ./infra/connect.sh {EXPERIMENT_NAME}
  2. Run setup: ./infra/setup-{EXPERIMENT_NAME}.sh
  3. Start training: cd experiments/{EXPERIMENT_NAME} && uv run python run.py
  4. Monitor: tail -f outputs/{EXPERIMENT_NAME}/log.txt
  5. When done: ./infra/destroy-{EXPERIMENT_NAME}.sh

🔗 Quick commands:

# Connect to VM
./infra/connect.sh {EXPERIMENT_NAME}

# Check VM status
pulumi stack output

# View logs
gcloud compute ssh {VM_NAME} --zone={VM_ZONE} --command="tail -f frontier-ready/outputs/{EXPERIMENT_NAME}/log.txt"

# Destroy when done
./infra/destroy-{EXPERIMENT_NAME}.sh

⚠️  IMPORTANT REMINDERS:
  - Destroy VM after experiment to stop charges
  - Preemptible VMs may restart (save checkpoints frequently)
  - Budget alert set at {BUDGET_ALERT} (monitor spend)
  - Data disk persists after VM destroyed (delete separately if not needed)
```

## Error Handling

### GCloud Not Installed
```bash
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    echo ""
    echo "Or use homebrew: brew install google-cloud-sdk"
    exit 1
fi
```

### Not Authenticated
```bash
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with GCP"
    echo "Run: gcloud auth login"
    echo "Then: gcloud auth application-default login"
    exit 1
fi
```

### No Project Set
```bash
if [ -z "${GCP_PROJECT}" ]; then
    echo "❌ No GCP project set"
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi
```

### Deployment Fails
```bash
if ! pulumi up --yes; then
    echo "❌ Deployment failed"
    echo ""
    echo "Common issues:"
    echo "  - Quota exceeded (check GCP quotas)"
    echo "  - Region doesn't have A100 availability"
    echo "  - Billing not enabled"
    echo ""
    echo "Check error above and retry"
    exit 1
fi
```

### VM Not Reachable
```bash
if ! gcloud compute ssh "${VM_NAME}" --zone="${VM_ZONE}" --command="echo test" 2>/dev/null; then
    echo "⚠️  VM not reachable via SSH"
    echo ""
    echo "Possible reasons:"
    echo "  - VM still booting (wait 2-3 minutes)"
    echo "  - Firewall blocking SSH"
    echo "  - SSH keys not configured"
    echo ""
    echo "Try: gcloud compute ssh ${VM_NAME} --zone=${VM_ZONE}"
fi
```

## Notes

- **Preemptible:** 70% cost savings but may be interrupted (good for experiments)
- **Quotas:** Check GCP quotas for A100 GPUs in your region
- **Regions:** A100 availability varies by region
- **Costs:** Charges accumulate while VM is running (destroy when done!)
- **Disks:** Boot disk auto-deleted with VM, data disk persists (delete manually)
- **Budget alerts:** Set alerts to avoid unexpected charges
- **Connection:** Use gcloud compute ssh for best compatibility
- **Setup:** Run setup script on VM to install dependencies
- **Monitoring:** Use `nvidia-smi` to check GPU utilization
- **Destruction:** Always destroy infrastructure after experiments complete

## Reference Files

- **Infra README:** `infra/README.md`
- **Pulumi example:** `infra/Pulumi.example.yaml`
- **Connect script:** `infra/connect.sh`
