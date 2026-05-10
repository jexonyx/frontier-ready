import * as gcp from "@pulumi/gcp";
import { InfraConfig, ResourceNames } from "../config/types";

export function createDataDisk(
  config: InfraConfig,
  names: ResourceNames
): gcp.compute.Disk {
  return new gcp.compute.Disk(names.dataDiskName, {
    name: names.dataDiskName,
    type: `projects/${config.gcpProject}/zones/${config.zone}/diskTypes/${config.dataDiskType}`,
    zone: config.zone,
    size: config.dataDiskSize,
    project: config.gcpProject,
    labels: {
      experiment: config.experimentName,
      environment: config.environment,
      managed_by: "pulumi",
      purpose: "data-storage",
    },
  }, {
    protect: false,
  });
}
