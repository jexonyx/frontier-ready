# Frontier Ready Infrastructure

GPU VM infrastructure for ML experiments using Pulumi + GCP.

## Setup

1. **Install dependencies:**
   ```bash
   cd infra
   npm install
   ```

2. **Configure GCP project:**

   **Option A: Environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your values
   export $(cat .env | xargs)
   ```

   **Option B: Pulumi config**
   ```bash
   pulumi config set gcpProject your-gcp-project-id
   pulumi config set experimentName nanogpt
   ```

3. **Initialize Pulumi stack:**
   ```bash
   pulumi stack init dev
   ```

## Usage

### Deploy a VM

```bash
# Using environment variables
export GCP_PROJECT=your-project-id
export EXPERIMENT_NAME=nanogpt
pulumi up

# Or override with command-line config
pulumi up --config experimentName=sae --config machineType=a2-highgpu-2g
```

### Check VM status

```bash
pulumi stack output
```

### SSH into VM

**Option 1: VS Code Remote SSH (Recommended)**
```bash
./connect.sh
# Follow the instructions to connect via VS Code
```

**Option 2: Simple Terminal SSH**
```bash
./connect-simple.sh
```

**Option 3: Manual SSH Config**

Add to `~/.ssh/config`:
```
Host frontier-ready-vm
    HostName INSTANCE_NAME
    User YOUR_USERNAME
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ProxyCommand gcloud compute start-iap-tunnel %h 22 --listen-on-stdin --project=frontier-ready --zone=ZONE --verbosity=warning
```

Replace:
- `YOUR_USERNAME` with `whoami` output
- `INSTANCE_NAME` with `pulumi stack output instanceName`
- `ZONE` with `pulumi stack output instanceZone`

Then connect via VS Code Remote-SSH or `ssh frontier-ready-vm`

**Option 4: Direct gcloud**
```bash
gcloud compute ssh $(pulumi stack output instanceName) \
  --zone=$(pulumi stack output instanceZone) \
  --project=frontier-ready
```

### Destroy VM

```bash
pulumi destroy
```

## Configuration

All configuration can be set via:
1. **Pulumi stack config** (`Pulumi.dev.yaml`)
2. **Environment variables** (see `.env.example`)
3. **Command-line flags** (`--config key=value`)

Priority: Command-line > Pulumi config > Environment variables > Defaults

### Available Configuration

| Config Key | Env Var | Default | Description |
|------------|---------|---------|-------------|
| `gcpProject` | `GCP_PROJECT` | *required* | GCP project ID |
| `experimentName` | `EXPERIMENT_NAME` | `default` | Experiment identifier |
| `environment` | `ENVIRONMENT` | `dev` | Environment (dev/prod) |
| `zone` | `GCP_ZONE` | `us-central1-a` | GCP zone |
| `machineType` | `MACHINE_TYPE` | `a2-highgpu-1g` | Machine type |
| `gpuType` | `GPU_TYPE` | `nvidia-tesla-a100` | GPU accelerator type |
| `gpuCount` | `GPU_COUNT` | `1` | Number of GPUs |
| `bootDiskSize` | `BOOT_DISK_SIZE` | `100` | Boot disk size (GB) |
| `bootDiskType` | `BOOT_DISK_TYPE` | `pd-standard` | Disk type |
| `imageFamily` | `IMAGE_FAMILY` | `pytorch-latest-gpu` | Base image family |
| `imageProject` | `IMAGE_PROJECT` | `deeplearning-platform-release` | Image project |
| `preemptible` | `PREEMPTIBLE` | `true` | Use preemptible instances |
| `network` | `NETWORK` | `default` | VPC network |
| `enableExternalIp` | `ENABLE_EXTERNAL_IP` | `true` | Assign external IP |

## Common Machine Types

| Machine Type | GPUs | GPU Memory | vCPUs | RAM | Approx Cost/hr (preemptible) |
|--------------|------|------------|-------|-----|------------------------------|
| `a2-highgpu-1g` | 1x A100 | 40GB | 12 | 85GB | ~$1.35 |
| `a2-highgpu-2g` | 2x A100 | 80GB | 24 | 170GB | ~$2.70 |
| `a2-ultragpu-1g` | 1x A100 | 80GB | 12 | 170GB | ~$1.60 |
| `g2-standard-4` | 1x L4 | 24GB | 4 | 16GB | ~$0.45 |

For H100 availability, check [GCP GPU regions](https://cloud.google.com/compute/docs/gpus).

## Examples

### Phase 1: NanoGPT Training (H100)

```bash
export GCP_PROJECT=your-project
export EXPERIMENT_NAME=phase1-nanogpt
export MACHINE_TYPE=a3-highgpu-8g  # 8x H100
export GPU_TYPE=nvidia-h100-80gb
export GPU_COUNT=1
export BOOT_DISK_SIZE=200
pulumi up
```

### Phase 5: SAE Training (Cost-optimized)

```bash
export EXPERIMENT_NAME=phase5-sae
export MACHINE_TYPE=g2-standard-4  # L4 GPU
export GPU_TYPE=nvidia-l4
export PREEMPTIBLE=true
pulumi up
```

## Tips

- **Preemptible instances** save ~70% cost but can be terminated
- Use **pd-standard** disks for cost, **pd-ssd** for performance
- Set `ENABLE_EXTERNAL_IP=false` if using Cloud NAT or private network
- VM automatically installs NVIDIA drivers via metadata flag
- Default startup script installs nvidia-container-toolkit for Docker

## Troubleshooting

### GPU quota errors

```bash
# Check quotas
gcloud compute regions describe us-central1 --format="value(quotas)"

# Request quota increase
# https://console.cloud.google.com/iam-admin/quotas
```

### SSH connection issues

```bash
# Ensure firewall allows SSH
gcloud compute firewall-rules list

# Check VM status
pulumi stack output
gcloud compute instances describe $(pulumi stack output instanceName) --zone=$(pulumi stack output instanceZone)
```
