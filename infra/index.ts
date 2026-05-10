import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";

// Configuration with environment variable fallbacks
const config = new pulumi.Config();

// GCP Project Configuration
const gcpProject = config.get("gcpProject") || process.env.GCP_PROJECT;
if (!gcpProject) {
    throw new Error("GCP project must be specified via config or GCP_PROJECT env var");
}

// Experiment Configuration
const experimentName = config.get("experimentName") || process.env.EXPERIMENT_NAME || "default";
const environment = config.get("environment") || process.env.ENVIRONMENT || "dev";

// Compute Configuration
const zone = config.get("zone") || process.env.GCP_ZONE || "us-central1-a";
const machineType = config.get("machineType") || process.env.MACHINE_TYPE || "a2-highgpu-1g";
const gpuType = config.get("gpuType") || process.env.GPU_TYPE || "nvidia-tesla-a100";
const gpuCount = config.getNumber("gpuCount") || Number(process.env.GPU_COUNT) || 1;

// Storage Configuration
const bootDiskSize = config.getNumber("bootDiskSize") || Number(process.env.BOOT_DISK_SIZE) || 100;
const bootDiskType = config.get("bootDiskType") || process.env.BOOT_DISK_TYPE || "pd-standard";
const imageFamily = config.get("imageFamily") || process.env.IMAGE_FAMILY || "pytorch-latest-gpu";
const imageProject = config.get("imageProject") || process.env.IMAGE_PROJECT || "deeplearning-platform-release";

// Cost Optimization
const preemptible = config.getBoolean("preemptible") ?? (process.env.PREEMPTIBLE === "true" ? true : false);

// Network Configuration
const network = config.get("network") || process.env.NETWORK || "default";
const enableExternalIp = config.getBoolean("enableExternalIp") ?? (process.env.ENABLE_EXTERNAL_IP === "false" ? false : true);

// Startup Script (optional)
const startupScript = config.get("startupScript") || process.env.STARTUP_SCRIPT || `#!/bin/bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

echo "GPU VM ready for ML experiments"
`;

// Construct resource names
const vmName = `${experimentName}-${environment}-vm`;

// Create the GPU VM instance
const instance = new gcp.compute.Instance(vmName, {
    name: vmName,
    machineType: machineType,
    zone: zone,
    project: gcpProject,

    // Boot disk configuration
    bootDisk: {
        initializeParams: {
            image: `projects/${imageProject}/global/images/family/${imageFamily}`,
            size: bootDiskSize,
            type: bootDiskType,
        },
        autoDelete: true,
    },

    // GPU configuration
    guestAccelerators: [{
        type: `projects/${gcpProject}/zones/${zone}/acceleratorTypes/${gpuType}`,
        count: gpuCount,
    }],

    // Scheduling: preemptible for cost savings
    scheduling: {
        preemptible: preemptible,
        automaticRestart: false,
        onHostMaintenance: "TERMINATE", // Required for GPU VMs
    },

    // Network configuration
    networkInterfaces: [{
        network: network,
        ...(enableExternalIp ? {
            accessConfigs: [{
                natIp: undefined, // Ephemeral IP
                networkTier: "STANDARD", // Use STANDARD tier for cost savings
            }],
        } : {}),
    }],

    // Metadata and startup script
    metadata: {
        "startup-script": startupScript,
        "install-nvidia-driver": "True", // GCP auto-installs NVIDIA drivers
    },

    // Service account with appropriate scopes
    serviceAccount: {
        scopes: [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring.write",
        ],
    },

    // Tags for firewall rules (optional)
    tags: ["ml-experiment", experimentName],

    // Labels for cost tracking
    labels: {
        experiment: experimentName,
        environment: environment,
        managed_by: "pulumi",
    },
}, {
    // Protect against accidental deletion if needed
    protect: false,
});

// Exports
export const instanceName = instance.name;
export const instanceZone = instance.zone;
export const internalIp = instance.networkInterfaces.apply(ni => ni[0].networkIp);
export const externalIp = enableExternalIp
    ? instance.networkInterfaces.apply(ni => ni[0].accessConfigs?.[0]?.natIp || "Pending")
    : pulumi.output("No external IP configured");
export const sshCommand = pulumi.interpolate`gcloud compute ssh ${instance.name} --zone=${zone} --project=${gcpProject}`;
export const machineConfig = pulumi.output({
    machineType: machineType,
    gpuType: gpuType,
    gpuCount: gpuCount,
    preemptible: preemptible,
    zone: zone,
});
