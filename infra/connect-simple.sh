#!/bin/bash
# Simple script to SSH into the VM using gcloud (bypasses IP entirely)

set -e

cd "$(dirname "$0")"

# Get Pulumi outputs
INSTANCE_NAME=$(pulumi stack output instanceName 2>/dev/null)
ZONE=$(pulumi stack output instanceZone 2>/dev/null)
PROJECT=$(pulumi config get gcpProject)

if [ -z "$INSTANCE_NAME" ]; then
    echo "No VM deployed. Run 'pulumi up' first."
    exit 1
fi

echo "Connecting to $INSTANCE_NAME in $ZONE..."
echo ""

# Use gcloud ssh (handles auth automatically)
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT"
