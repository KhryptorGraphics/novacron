#!/bin/bash
# Script to create an Ubuntu 24.04 VM using the NovaCron API directly

set -e

# Configuration
VM_NAME=${1:-"ubuntu-24-04-vm"}
API_ENDPOINT="http://localhost:8090"
IMAGE_PATH="/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2"

echo "Creating Ubuntu 24.04 VM: $VM_NAME"
echo "Using API endpoint: $API_ENDPOINT"
echo "Using image: $IMAGE_PATH"

# Create a JSON payload for the VM creation request
JSON_PAYLOAD=$(cat <<EOF
{
  "name": "$VM_NAME",
  "memory_mb": 2048,
  "vcpus": 2,
  "disk_mb": 20480,
  "image": "$IMAGE_PATH",
  "network_config": {
    "interfaces": [
      {
        "network_id": "default"
      }
    ]
  },
  "tags": ["ubuntu", "24.04", "lts"]
}
EOF
)

# Make the API request
echo "Sending VM creation request..."
echo "$JSON_PAYLOAD" | jq .

echo ""
echo "Sending request to $API_ENDPOINT/api/v1/vms"
RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "$JSON_PAYLOAD" "$API_ENDPOINT/api/v1/vms")

echo ""
echo "Response:"
echo "$RESPONSE" | jq .

if [[ "$RESPONSE" == *"error"* ]]; then
  echo ""
  echo "Error creating VM. Check the API logs for more details."
  exit 1
fi

echo ""
echo "VM creation request sent successfully!"
echo "Check the API logs for more details."

exit 0
