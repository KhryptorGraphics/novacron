#!/bin/bash
# Dashboard script for monitoring Ubuntu 24.04 VMs in NovaCron

# Configuration
API_ENDPOINT="http://localhost:8080"
REFRESH_INTERVAL=10  # seconds
LOG_FILE="/var/log/novacron/ubuntu_24_04_dashboard.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Function to print status messages
log_message() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] - $1" >> "$LOG_FILE"
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

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

log_message "Starting Ubuntu 24.04 VM Dashboard"

# Function to clear screen and display header
display_header() {
    clear
    echo -e "${WHITE}=======================================================================${NC}"
    echo -e "${WHITE}                 NOVACRON UBUNTU 24.04 VM DASHBOARD                    ${NC}"
    echo -e "${WHITE}=======================================================================${NC}"
    echo -e "${BLUE}Date: $(date)${NC}"
    echo -e "${BLUE}API Endpoint: $API_ENDPOINT${NC}"
    echo -e "${BLUE}Refresh Interval: $REFRESH_INTERVAL seconds${NC}"
    echo -e "${WHITE}=======================================================================${NC}"
    echo ""
}

# Function to display service status
display_service_status() {
    echo -e "${WHITE}SERVICE STATUS${NC}"
    echo -e "${WHITE}----------------------------------------------------------------------${NC}"
    
    # Check hypervisor service
    hypervisor_status=$(systemctl is-active novacron-hypervisor.service)
    if [ "$hypervisor_status" = "active" ]; then
        echo -e "${GREEN}Hypervisor Service: RUNNING${NC}"
    else
        echo -e "${RED}Hypervisor Service: NOT RUNNING${NC}"
    fi
    
    # Check API service
    api_status=$(systemctl is-active novacron-api.service)
    if [ "$api_status" = "active" ]; then
        echo -e "${GREEN}API Service: RUNNING${NC}"
    else
        echo -e "${RED}API Service: NOT RUNNING${NC}"
    fi
    
    # Check monitor service
    monitor_status=$(systemctl is-active novacron-ubuntu-24-04-monitor.service)
    if [ "$monitor_status" = "active" ]; then
        echo -e "${GREEN}Monitor Service: RUNNING${NC}"
    else
        echo -e "${RED}Monitor Service: NOT RUNNING${NC}"
    fi
    
    echo ""
}

# Function to display Ubuntu 24.04 VM count
display_vm_count() {
    echo -e "${WHITE}UBUNTU 24.04 VM STATISTICS${NC}"
    echo -e "${WHITE}----------------------------------------------------------------------${NC}"
    
    # Get all VMs
    VM_LIST=$(make_request "GET" "/api/v1/vms")
    
    # Count Ubuntu 24.04 VMs
    UBUNTU_24_04_VMS=$(echo "$VM_LIST" | jq '[.[] | select(.spec.image | contains("ubuntu-24.04"))]')
    VM_COUNT=$(echo "$UBUNTU_24_04_VMS" | jq 'length')
    
    # Count by state
    RUNNING_COUNT=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | select(.state == "running")] | length')
    STOPPED_COUNT=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | select(.state == "stopped")] | length')
    CREATING_COUNT=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | select(.state == "creating")] | length')
    ERROR_COUNT=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | select(.state == "error")] | length')
    
    echo -e "${CYAN}Total Ubuntu 24.04 VMs: $VM_COUNT${NC}"
    echo -e "${GREEN}Running: $RUNNING_COUNT${NC}"
    echo -e "${YELLOW}Stopped: $STOPPED_COUNT${NC}"
    echo -e "${BLUE}Creating: $CREATING_COUNT${NC}"
    echo -e "${RED}Error: $ERROR_COUNT${NC}"
    
    echo ""
}

# Function to display VM list
display_vm_list() {
    echo -e "${WHITE}UBUNTU 24.04 VM LIST${NC}"
    echo -e "${WHITE}----------------------------------------------------------------------${NC}"
    echo -e "${WHITE}ID                                      NAME                  STATE      VCPU  MEM(MB)  DISK(MB)${NC}"
    echo -e "${WHITE}----------------------------------------------------------------------${NC}"
    
    # Get all VMs
    VM_LIST=$(make_request "GET" "/api/v1/vms")
    
    # Filter Ubuntu 24.04 VMs
    UBUNTU_24_04_VMS=$(echo "$VM_LIST" | jq '[.[] | select(.spec.image | contains("ubuntu-24.04"))]')
    
    # Display VM list
    echo "$UBUNTU_24_04_VMS" | jq -r '.[] | "\(.id)  \(.name | .[0:20] | %-20s)  \(.state | %-10s)  \(.spec.vcpus)     \(.spec.memory_mb)    \(.spec.disk_mb)"'
    
    echo ""
}

# Function to display resource usage
display_resource_usage() {
    echo -e "${WHITE}RESOURCE USAGE${NC}"
    echo -e "${WHITE}----------------------------------------------------------------------${NC}"
    
    # Get all VMs
    VM_LIST=$(make_request "GET" "/api/v1/vms")
    
    # Filter Ubuntu 24.04 VMs
    UBUNTU_24_04_VMS=$(echo "$VM_LIST" | jq '[.[] | select(.spec.image | contains("ubuntu-24.04"))]')
    
    # Calculate total resources
    TOTAL_VCPUS=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | .spec.vcpus] | add')
    TOTAL_MEMORY=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | .spec.memory_mb] | add')
    TOTAL_DISK=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | .spec.disk_mb] | add')
    
    # Calculate resources for running VMs
    RUNNING_VCPUS=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | select(.state == "running") | .spec.vcpus] | add')
    RUNNING_MEMORY=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | select(.state == "running") | .spec.memory_mb] | add')
    RUNNING_DISK=$(echo "$UBUNTU_24_04_VMS" | jq '[.[] | select(.state == "running") | .spec.disk_mb] | add')
    
    # Handle null values
    TOTAL_VCPUS=${TOTAL_VCPUS:-0}
    TOTAL_MEMORY=${TOTAL_MEMORY:-0}
    TOTAL_DISK=${TOTAL_DISK:-0}
    RUNNING_VCPUS=${RUNNING_VCPUS:-0}
    RUNNING_MEMORY=${RUNNING_MEMORY:-0}
    RUNNING_DISK=${RUNNING_DISK:-0}
    
    echo -e "${CYAN}Total Resources Allocated:${NC}"
    echo -e "  vCPUs: $TOTAL_VCPUS"
    echo -e "  Memory: $(echo "$TOTAL_MEMORY / 1024" | bc -l | xargs printf "%.2f") GB"
    echo -e "  Disk: $(echo "$TOTAL_DISK / 1024" | bc -l | xargs printf "%.2f") GB"
    
    echo -e "${GREEN}Resources Used by Running VMs:${NC}"
    echo -e "  vCPUs: $RUNNING_VCPUS"
    echo -e "  Memory: $(echo "$RUNNING_MEMORY / 1024" | bc -l | xargs printf "%.2f") GB"
    echo -e "  Disk: $(echo "$RUNNING_DISK / 1024" | bc -l | xargs printf "%.2f") GB"
    
    echo ""
}

# Function to display recent events
display_recent_events() {
    echo -e "${WHITE}RECENT EVENTS${NC}"
    echo -e "${WHITE}----------------------------------------------------------------------${NC}"
    
    # Get recent events from log file
    if [ -f "$LOG_FILE" ]; then
        tail -n 5 "$LOG_FILE"
    else
        echo "No events logged yet"
    fi
    
    echo ""
}

# Function to display footer
display_footer() {
    echo -e "${WHITE}=======================================================================${NC}"
    echo -e "${WHITE}Press Ctrl+C to exit${NC}"
    echo -e "${WHITE}=======================================================================${NC}"
}

# Main loop
trap "echo -e '\nExiting dashboard...'; exit 0" INT
while true; do
    display_header
    display_service_status
    display_vm_count
    display_vm_list
    display_resource_usage
    display_recent_events
    display_footer
    
    log_message "Dashboard refreshed"
    sleep $REFRESH_INTERVAL
done
