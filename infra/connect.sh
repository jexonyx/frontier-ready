#!/bin/bash
# Script to connect to the deployed VM via VS Code

set -e

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Fetching VM information...${NC}"

# Get Pulumi outputs
INSTANCE_NAME=$(pulumi stack output instanceName 2>/dev/null)
ZONE=$(pulumi stack output instanceZone 2>/dev/null)
EXTERNAL_IP=$(pulumi stack output externalIp 2>/dev/null)
PROJECT=$(pulumi config get gcpProject)

if [ -z "$INSTANCE_NAME" ]; then
    echo -e "${YELLOW}No VM deployed. Run 'pulumi up' first.${NC}"
    exit 1
fi

echo -e "${GREEN}VM Details:${NC}"
echo "  Name: $INSTANCE_NAME"
echo "  Zone: $ZONE"
echo "  External IP: $EXTERNAL_IP"
echo "  Project: $PROJECT"
echo ""

# Update SSH config for VS Code
SSH_CONFIG="$HOME/.ssh/config"
SSH_HOST_ENTRY="frontier-ready-vm"

echo -e "${BLUE}Updating SSH config for VS Code...${NC}"

# Remove existing entry if present
if grep -q "Host $SSH_HOST_ENTRY" "$SSH_CONFIG" 2>/dev/null; then
    # Remove old entry (from Host line to next Host line or EOF)
    sed -i.bak "/^Host $SSH_HOST_ENTRY$/,/^Host /{ /^Host $SSH_HOST_ENTRY$/d; /^Host /!d; }" "$SSH_CONFIG"
fi

# Add new entry
cat >> "$SSH_CONFIG" << EOF

# Frontier Ready ML VM (auto-generated)
Host $SSH_HOST_ENTRY
    HostName $INSTANCE_NAME
    User $(whoami)
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ProxyCommand gcloud compute start-iap-tunnel %h 22 --listen-on-stdin --project=$PROJECT --zone=$ZONE --verbosity=warning

EOF

echo -e "${GREEN}✓ SSH config updated${NC}"
echo ""
echo -e "${BLUE}Connection Options:${NC}"
echo ""
echo -e "${GREEN}1. VS Code Remote SSH:${NC}"
echo "   - Press Cmd+Shift+P"
echo "   - Type 'Remote-SSH: Connect to Host'"
echo "   - Select: $SSH_HOST_ENTRY"
echo ""
echo -e "${GREEN}2. Terminal SSH:${NC}"
echo "   ssh $SSH_HOST_ENTRY"
echo ""
echo -e "${GREEN}3. VS Code CLI (if installed):${NC}"
echo "   code --remote ssh-remote+$SSH_HOST_ENTRY /workspace"
echo ""
echo -e "${YELLOW}Note: First connection may take a moment while gcloud authenticates${NC}"
