#!/bin/bash
set -e

# Format and mount data disk if not already formatted
MOUNT_POINT="/data"

# Detect the data disk - try common device names
# On older instances: /dev/sdb
# On NVMe instances (G2, etc.): /dev/nvme0n2 (nvme0n1 is boot disk)
if [ -b "/dev/nvme0n2" ]; then
    DATA_DISK="/dev/nvme0n2"
elif [ -b "/dev/sdb" ]; then
    DATA_DISK="/dev/sdb"
else
    echo "Warning: No data disk found (checked /dev/sdb and /dev/nvme0n2)"
    exit 0
fi

echo "Data disk found at $DATA_DISK"

# Check if disk is already formatted
if ! blkid "$DATA_DISK"; then
    echo "Formatting data disk..."
    mkfs.ext4 -F "$DATA_DISK"
fi

# Create mount point
mkdir -p "$MOUNT_POINT"

# Mount the disk
if ! mountpoint -q "$MOUNT_POINT"; then
    echo "Mounting data disk..."
    mount "$DATA_DISK" "$MOUNT_POINT"

    # Add to fstab for automatic mounting on reboot
    DISK_UUID=$(blkid -s UUID -o value "$DATA_DISK")
    if ! grep -q "$DISK_UUID" /etc/fstab; then
        echo "UUID=$DISK_UUID $MOUNT_POINT ext4 discard,defaults,nofail 0 2" >> /etc/fstab
    fi
fi

# Set ownership to the default user (first user in /home)
DEFAULT_USER=$(ls /home | head -n 1)
if [ -n "$DEFAULT_USER" ]; then
    chown -R "$DEFAULT_USER:$DEFAULT_USER" "$MOUNT_POINT"
fi
chmod 755 "$MOUNT_POINT"

echo "Data disk mounted at $MOUNT_POINT"

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

echo "GPU VM ready for ML experiments"
echo "Data disk available at $MOUNT_POINT"
