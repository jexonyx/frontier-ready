import * as pulumi from "@pulumi/pulumi";
import { loadConfig, getResourceNames } from "./config";
import { createDataDisk, createInstance } from "./resources";
import { createOutputs } from "./outputs";

// Load configuration
const config = loadConfig();
const names = getResourceNames(config);

// Create resources in dependency order
const dataDisk = createDataDisk(config, names);
const instance = createInstance(config, names, dataDisk);

// Create and export outputs
const outputs = createOutputs(config, instance, dataDisk);

export const instanceName = outputs.instanceName;
export const instanceZone = outputs.instanceZone;
export const internalIp = outputs.internalIp;
export const externalIp = outputs.externalIp;
export const sshCommand = outputs.sshCommand;
export const dataDiskName = outputs.dataDiskName;
export const dataDiskSize = outputs.dataDiskSize;
export const dataDiskType = outputs.dataDiskType;
export const dataMountPoint = outputs.dataMountPoint;
export const machineConfig = outputs.machineConfig;
