import * as pulumi from "@pulumi/pulumi";
import * as fs from "fs";
import * as path from "path";
import { InfraConfig, ResourceNames } from "./types";

export function loadConfig(): InfraConfig {
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
  const dataDiskSize = config.getNumber("dataDiskSize") || Number(process.env.DATA_DISK_SIZE) || 200;
  const dataDiskType = config.get("dataDiskType") || process.env.DATA_DISK_TYPE || "pd-ssd";
  const imageFamily = config.get("imageFamily") || process.env.IMAGE_FAMILY || "pytorch-latest-gpu";
  const imageProject = config.get("imageProject") || process.env.IMAGE_PROJECT || "deeplearning-platform-release";

  // Cost Optimization
  const preemptible = config.getBoolean("preemptible") ?? (process.env.PREEMPTIBLE === "true" ? true : false);

  // Network Configuration
  const network = config.get("network") || process.env.NETWORK || "default";
  const enableExternalIp = config.getBoolean("enableExternalIp") ?? (process.env.ENABLE_EXTERNAL_IP === "false" ? false : true);

  // Startup Script (optional)
  const startupScript = config.get("startupScript") || process.env.STARTUP_SCRIPT || loadDefaultStartupScript();

  return {
    gcpProject,
    experimentName,
    environment,
    zone,
    machineType,
    gpuType,
    gpuCount,
    bootDiskSize,
    bootDiskType,
    dataDiskSize,
    dataDiskType,
    imageFamily,
    imageProject,
    preemptible,
    network,
    enableExternalIp,
    startupScript,
  };
}

function loadDefaultStartupScript(): string {
  const scriptPath = path.join(__dirname, "../scripts/startup.sh");
  return fs.readFileSync(scriptPath, "utf-8");
}

export function getResourceNames(config: InfraConfig): ResourceNames {
  return {
    vmName: `${config.experimentName}-${config.environment}-vm`,
    dataDiskName: `${config.experimentName}-${config.environment}-data`,
  };
}
