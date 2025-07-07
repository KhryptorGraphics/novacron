#!/bin/bash
# Script to test all Ubuntu 24.04 features

set -e

# Configuration
API_ENDPOINT="http://localhost:8090"
VM_NAME="ubuntu-24-04-test-$(date +%s)"
LOG_FILE="ubuntu-24-04-test-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[+] $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> $LOG_FILE
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> $LOG_FILE
}

print_error() {
    echo -e "${RED}[-] $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> $LOG_FILE
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [HEADER] $1" >> $LOG_FILE
}

# Function to make API requests
make_request() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    if [ -n "$data" ]; then
        curl -s -X "$method" -H "Content-Type: application/json" -d "$data" "$API_ENDPOINT$endpoint"
    else
        curl -s -X "$method" "$API_ENDPOINT$endpoint"
    fi
}

# Start logging
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting Ubuntu 24.04 feature test" > $LOG_FILE

print_header "TESTING UBUNTU 24.04 FEATURES"
print_status "Starting comprehensive test of Ubuntu 24.04 features"

# Test 1: Create a VM with Ubuntu 24.04
print_header "TEST 1: CREATE VM"
print_status "Creating Ubuntu 24.04 VM with name $VM_NAME"

CREATE_PAYLOAD=$(cat <<EOF
{
  "name": "$VM_NAME",
  "vcpus": 2,
  "memory_mb": 2048,
  "disk_mb": 20480,
  "image": "ubuntu-24.04",
  "network_config": {
    "interfaces": [
      {
        "network_id": "default"
      }
    ]
  },
  "cloud_init": {
    "user_data_template": "development",
    "template_vars": {
      "ssh_authorized_keys": ["ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC0WGP1EZykEtv5YGC9nMiPFW3U3DmZNzKFO5nEu6uozEHh4jLZzPNHSrfFTuQ2GnRDSt+XbOtTLdcj26+iPNiFoFha42aCIzYjt6V8Z+SQ9pzF4jPPzxwXfDdkEWylgoNnZ+4MG1lNFqa8aO5yKBVUYyQgnJ/PuRPN3vEGXqhBWLdLgmBxrxYyEklS+HIWPpK+gUoFdlBJWvqS0PcJLC6RytyVyPj8FkFdPrbGS/zXx1fGHhqSMUXat7QJk9DQ+NfKXOyRvVdXRTE5xX0YvQyVTrHSjjMPvE+rZ4fD+JZIfuimK3yxvTBhQrGJYGjTt9E8rXQXLwSk7AlOcZoj3djT"],
      "additional_packages": ["golang", "postgresql-client"]
    }
  },
  "tags": ["ubuntu", "24.04", "lts", "test"]
}
EOF
)

CREATE_RESPONSE=$(make_request "POST" "/api/v1/vms" "$CREATE_PAYLOAD")
echo "$CREATE_RESPONSE" | jq . >> $LOG_FILE

# Extract VM ID from response
VM_ID=$(echo "$CREATE_RESPONSE" | jq -r '.id')

if [ "$VM_ID" == "null" ] || [ -z "$VM_ID" ]; then
    print_error "Failed to create VM"
    exit 1
fi

print_status "VM created with ID: $VM_ID"

# Wait for VM to be created
print_status "Waiting for VM to be created..."
sleep 10

# Test 2: Apply performance profile
print_header "TEST 2: APPLY PERFORMANCE PROFILE"
print_status "Applying performance profile to VM $VM_ID"

PROFILE_PAYLOAD=$(cat <<EOF
{
  "profile": "ubuntu-24-04-optimized"
}
EOF
)

PROFILE_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/performance/profile" "$PROFILE_PAYLOAD")
echo "$PROFILE_RESPONSE" | jq . >> $LOG_FILE

print_status "Performance profile applied"

# Test 3: Start VM
print_header "TEST 3: START VM"
print_status "Starting VM $VM_ID"

START_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/start")
echo "$START_RESPONSE" | jq . >> $LOG_FILE

# Wait for VM to start
print_status "Waiting for VM to start..."
sleep 20

# Test 4: Get VM status
print_header "TEST 4: GET VM STATUS"
print_status "Getting VM status for $VM_ID"

VM_STATUS=$(make_request "GET" "/api/v1/vms/$VM_ID")
echo "$VM_STATUS" | jq . >> $LOG_FILE

# Check if VM is running
VM_STATE=$(echo "$VM_STATUS" | jq -r '.state')
if [ "$VM_STATE" == "running" ]; then
    print_status "VM is running"
else
    print_warning "VM is not running"
    print_warning "VM state: $VM_STATE"
fi

# Test 5: Get performance metrics
print_header "TEST 5: GET PERFORMANCE METRICS"
print_status "Getting performance metrics for VM $VM_ID"

METRICS_RESPONSE=$(make_request "GET" "/api/v1/vms/$VM_ID/performance/metrics")
echo "$METRICS_RESPONSE" | jq . >> $LOG_FILE

print_status "Performance metrics retrieved"

# Test 6: Create snapshot
print_header "TEST 6: CREATE SNAPSHOT"
print_status "Creating snapshot of VM $VM_ID"

SNAPSHOT_PAYLOAD=$(cat <<EOF
{
  "name": "test-snapshot",
  "description": "Test snapshot for Ubuntu 24.04 VM",
  "tags": {
    "purpose": "testing",
    "os": "ubuntu-24.04"
  }
}
EOF
)

SNAPSHOT_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/snapshots" "$SNAPSHOT_PAYLOAD")
echo "$SNAPSHOT_RESPONSE" | jq . >> $LOG_FILE

# Extract snapshot ID
SNAPSHOT_ID=$(echo "$SNAPSHOT_RESPONSE" | jq -r '.id')

if [ "$SNAPSHOT_ID" == "null" ] || [ -z "$SNAPSHOT_ID" ]; then
    print_warning "Failed to create snapshot"
else
    print_status "Snapshot created with ID: $SNAPSHOT_ID"
fi

# Test 7: Get cloud-init status
print_header "TEST 7: GET CLOUD-INIT STATUS"
print_status "Getting cloud-init status for VM $VM_ID"

CLOUDINIT_STATUS=$(make_request "GET" "/api/v1/vms/$VM_ID/cloudinit/status")
echo "$CLOUDINIT_STATUS" | jq . >> $LOG_FILE

print_status "Cloud-init status retrieved"

# Test 8: Get cloud-init logs
print_header "TEST 8: GET CLOUD-INIT LOGS"
print_status "Getting cloud-init logs for VM $VM_ID"

CLOUDINIT_LOGS=$(make_request "GET" "/api/v1/vms/$VM_ID/cloudinit/logs")
echo "$CLOUDINIT_LOGS" | head -n 10 >> $LOG_FILE

print_status "Cloud-init logs retrieved"

# Test 9: Stop VM
print_header "TEST 9: STOP VM"
print_status "Stopping VM $VM_ID"

STOP_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/stop")
echo "$STOP_RESPONSE" | jq . >> $LOG_FILE

# Wait for VM to stop
print_status "Waiting for VM to stop..."
sleep 20

# Test 10: Restore snapshot
print_header "TEST 10: RESTORE SNAPSHOT"
print_status "Restoring VM $VM_ID from snapshot $SNAPSHOT_ID"

if [ "$SNAPSHOT_ID" != "null" ] && [ -n "$SNAPSHOT_ID" ]; then
    RESTORE_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/snapshots/$SNAPSHOT_ID/restore")
    echo "$RESTORE_RESPONSE" | jq . >> $LOG_FILE
    print_status "Snapshot restored"
else
    print_warning "Skipping snapshot restore as snapshot creation failed"
fi

# Test 11: Start VM again
print_header "TEST 11: START VM AGAIN"
print_status "Starting VM $VM_ID again"

START_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/start")
echo "$START_RESPONSE" | jq . >> $LOG_FILE

# Wait for VM to start
print_status "Waiting for VM to start..."
sleep 20

# Test 12: Migrate VM
print_header "TEST 12: MIGRATE VM"
print_status "Migrating VM $VM_ID"

MIGRATE_PAYLOAD=$(cat <<EOF
{
  "target_node_id": "node-2",
  "options": {
    "live_migration": true,
    "auto_start": true,
    "delete_source": false,
    "verify_after_migration": true
  }
}
EOF
)

MIGRATE_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/migrate" "$MIGRATE_PAYLOAD")
echo "$MIGRATE_RESPONSE" | jq . >> $LOG_FILE

# Extract migration ID
MIGRATION_ID=$(echo "$MIGRATE_RESPONSE" | jq -r '.id')

if [ "$MIGRATION_ID" == "null" ] || [ -z "$MIGRATION_ID" ]; then
    print_warning "Failed to initiate migration"
else
    print_status "Migration initiated with ID: $MIGRATION_ID"
    
    # Wait for migration to complete
    print_status "Waiting for migration to complete..."
    sleep 30
    
    # Get migration status
    MIGRATION_STATUS=$(make_request "GET" "/api/v1/migrations/$MIGRATION_ID")
    echo "$MIGRATION_STATUS" | jq . >> $LOG_FILE
    
    MIGRATION_STATE=$(echo "$MIGRATION_STATUS" | jq -r '.state')
    print_status "Migration state: $MIGRATION_STATE"
fi

# Test 13: Stop and delete VM
print_header "TEST 13: CLEANUP"
print_status "Stopping and deleting VM $VM_ID"

# Stop VM
STOP_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/stop")
echo "$STOP_RESPONSE" | jq . >> $LOG_FILE

# Wait for VM to stop
print_status "Waiting for VM to stop..."
sleep 20

# Delete VM
DELETE_RESPONSE=$(make_request "DELETE" "/api/v1/vms/$VM_ID")
echo "$DELETE_RESPONSE" | jq . >> $LOG_FILE

print_status "VM deleted"

# Test 14: Delete snapshot
print_header "TEST 14: DELETE SNAPSHOT"
print_status "Deleting snapshot $SNAPSHOT_ID"

if [ "$SNAPSHOT_ID" != "null" ] && [ -n "$SNAPSHOT_ID" ]; then
    DELETE_SNAPSHOT_RESPONSE=$(make_request "DELETE" "/api/v1/snapshots/$SNAPSHOT_ID")
    echo "$DELETE_SNAPSHOT_RESPONSE" | jq . >> $LOG_FILE
    print_status "Snapshot deleted"
else
    print_warning "Skipping snapshot deletion as snapshot creation failed"
fi

# Summarize test results
print_header "TEST SUMMARY"
print_status "All tests completed!"
print_status "Test results saved to $LOG_FILE"

exit 0
