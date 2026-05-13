import * as pulumi from "@pulumi/pulumi";
import * as fs from "fs";
import * as path from "path";
import { InfraConfig, ResourceNames } from "./types";

export function loadConfig(): InfraConfig {
  // Configuration with environment variable fallbacks
  const config = new pulumi.Config();

  // Helper function to get required config
  const getRequired = (key: string, envVar?: string): string => {
    const value = config.get(key) || (envVar ? process.env[envVar] : undefined);
    if (!value) {
      throw new Error(`Required configuration '${key}' must be specified via Pulumi config${envVar ? ` or ${envVar} env var` : ''}`);
    }
    return value;
  };

  const getRequiredNumber = (key: string, envVar?: string): number => {
    const value = config.getNumber(key) ?? (envVar && process.env[envVar] ? Number(process.env[envVar]) : undefined);
    if (value === undefined || isNaN(value)) {
      throw new Error(`Required configuration '${key}' must be specified as a number via Pulumi config${envVar ? ` or ${envVar} env var` : ''}`);
    }
    return value;
  };

  const getRequiredBoolean = (key: string, envVar?: string): boolean => {
    const value = config.getBoolean(key) ?? (envVar && process.env[envVar] ? process.env[envVar] === "true" : undefined);
    if (value === undefined) {
      throw new Error(`Required configuration '${key}' must be specified as a boolean via Pulumi config${envVar ? ` or ${envVar} env var` : ''}`);
    }
    return value;
  };

  // GCP Project Configuration
  const gcpProject = getRequired("gcpProject", "GCP_PROJECT");

  // Experiment Configuration
  const experimentName = getRequired("experimentName", "EXPERIMENT_NAME");
  const environment = getRequired("environment", "ENVIRONMENT");

  // Compute Configuration
  const zone = getRequired("zone", "GCP_ZONE");
  const machineType = getRequired("machineType", "MACHINE_TYPE");
  const gpuType = getRequired("gpuType", "GPU_TYPE");
  const gpuCount = getRequiredNumber("gpuCount", "GPU_COUNT");

  // Storage Configuration
  const bootDiskSize = getRequiredNumber("bootDiskSize", "BOOT_DISK_SIZE");
  const bootDiskType = getRequired("bootDiskType", "BOOT_DISK_TYPE");
  const dataDiskSize = getRequiredNumber("dataDiskSize", "DATA_DISK_SIZE");
  const dataDiskType = getRequired("dataDiskType", "DATA_DISK_TYPE");
  const imageFamily = getRequired("imageFamily", "IMAGE_FAMILY");
  const imageProject = getRequired("imageProject", "IMAGE_PROJECT");

  // Cost Optimization
  const preemptible = getRequiredBoolean("preemptible", "PREEMPTIBLE");

  // Network Configuration
  const network = getRequired("network", "NETWORK");
  const enableExternalIp = getRequiredBoolean("enableExternalIp", "ENABLE_EXTERNAL_IP");

  // Startup Script (optional - falls back to default script file)
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
