import * as pulumi from "@pulumi/pulumi";
import * as gcp from "@pulumi/gcp";
import { InfraConfig } from "./config/types";

export function createOutputs(
  config: InfraConfig,
  instance: gcp.compute.Instance,
  dataDisk: gcp.compute.Disk
) {
  // Use local variable names to avoid shadowing
  const diskSizeConfig = config.dataDiskSize;
  const diskTypeConfig = config.dataDiskType;

  return {
    instanceName: instance.name,
    instanceZone: instance.zone,
    internalIp: instance.networkInterfaces.apply(ni => ni[0].networkIp),
    externalIp: config.enableExternalIp
      ? instance.networkInterfaces.apply(ni => ni[0].accessConfigs?.[0]?.natIp || "Pending")
      : pulumi.output("No external IP configured"),
    sshCommand: pulumi.interpolate`gcloud compute ssh ${instance.name} --zone=${config.zone} --project=${config.gcpProject}`,

    dataDiskName: dataDisk.name,
    dataDiskSize: pulumi.output(diskSizeConfig),  // No shadowing - uses local var
    dataDiskType: pulumi.output(diskTypeConfig),  // No shadowing - uses local var
    dataMountPoint: pulumi.output("/data"),

    machineConfig: pulumi.output({
      machineType: config.machineType,
      gpuType: config.gpuType,
      gpuCount: config.gpuCount,
      bootDiskSize: config.bootDiskSize,
      bootDiskType: config.bootDiskType,
      dataDiskSize: diskSizeConfig,  // Clear reference to config value
      dataDiskType: diskTypeConfig,  // Clear reference to config value
      preemptible: config.preemptible,
      zone: config.zone,
    }),
  };
}
