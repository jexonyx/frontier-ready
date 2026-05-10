export interface InfraConfig {
  gcpProject: string;
  experimentName: string;
  environment: string;
  zone: string;
  machineType: string;
  gpuType: string;
  gpuCount: number;
  bootDiskSize: number;
  bootDiskType: string;
  dataDiskSize: number;
  dataDiskType: string;
  imageFamily: string;
  imageProject: string;
  preemptible: boolean;
  network: string;
  enableExternalIp: boolean;
  startupScript: string;
}

export interface ResourceNames {
  vmName: string;
  dataDiskName: string;
}
