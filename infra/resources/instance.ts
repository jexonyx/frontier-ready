import * as gcp from "@pulumi/gcp";
import { InfraConfig, ResourceNames } from "../config/types";

export function createInstance(
  config: InfraConfig,
  names: ResourceNames,
  dataDisk: gcp.compute.Disk
): gcp.compute.Instance {
  return new gcp.compute.Instance(names.vmName, {
    name: names.vmName,
    machineType: config.machineType,
    zone: config.zone,
    project: config.gcpProject,

    // Boot disk configuration
    bootDisk: {
      initializeParams: {
        image: `projects/${config.imageProject}/global/images/family/${config.imageFamily}`,
        size: config.bootDiskSize,
        type: config.bootDiskType,
      },
      autoDelete: true,
    },

    // Attach the data disk
    attachedDisks: [{
      source: dataDisk.id,
      mode: "READ_WRITE",
      deviceName: "data-disk",
    }],

    // GPU configuration
    guestAccelerators: [{
      type: `projects/${config.gcpProject}/zones/${config.zone}/acceleratorTypes/${config.gpuType}`,
      count: config.gpuCount,
    }],

    // Scheduling: preemptible for cost savings
    scheduling: {
      preemptible: config.preemptible,
      automaticRestart: false,
      onHostMaintenance: "TERMINATE", // Required for GPU VMs
    },

    // Network configuration
    networkInterfaces: [{
      network: config.network,
      ...(config.enableExternalIp ? {
        accessConfigs: [{
          natIp: undefined, // Ephemeral IP
          networkTier: "STANDARD", // Use STANDARD tier for cost savings
        }],
      } : {}),
    }],

    // Metadata and startup script
    metadata: {
      "startup-script": config.startupScript,
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
    tags: ["ml-experiment", config.experimentName],

    // Labels for cost tracking
    labels: {
      experiment: config.experimentName,
      environment: config.environment,
      managed_by: "pulumi",
    },
  }, {
    // Protect against accidental deletion if needed
    protect: false,
  });
}
