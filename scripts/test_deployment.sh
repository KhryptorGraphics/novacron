#!/bin/bash
# Script to test the deployment of Ubuntu 24.04 support in NovaCron

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[+] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

print_error() {
    echo -e "${RED}[-] $1${NC}"
}

# Step 1: Test the API service
print_status "Step 1: Testing the API service..."
print_status "Starting the mock hypervisor service..."
cd scripts && python3 mock_hypervisor.py &
HYPERVISOR_PID=$!
cd ..

# Wait for the hypervisor to start
sleep 3

print_status "Starting the API service..."
cd backend/services/api && python3 main.py --host 0.0.0.0 --port 8090 --config ../../../config/novacron/api.yaml --debug &
API_PID=$!
cd ../../..

# Wait for the API to start
sleep 3

# Step 2: Test VM creation
print_status "Step 2: Testing VM creation..."
./scripts/create_ubuntu_24_04_vm_api.sh test-vm-1

# Step 3: Test VM lifecycle
print_status "Step 3: Testing VM lifecycle..."
./scripts/test_ubuntu_24_04_lifecycle.sh

# Step 4: Test the frontend (if available)
print_status "Step 4: Testing the frontend..."
if [ -d "frontend/node_modules" ]; then
    print_status "Starting the frontend..."
    cd frontend && npm run dev &
    FRONTEND_PID=$!
    cd ..
    
    print_status "Frontend should be available at http://localhost:3000"
    print_status "Please open this URL in your browser to test the UI"
    print_status "You should be able to create a new VM with Ubuntu 24.04"
else
    print_warning "Frontend dependencies not installed, skipping frontend test"
fi

# Wait for user input
read -p "Press Enter to stop the services and complete the test..."

# Clean up
print_status "Stopping services..."
if [ -n "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID || true
fi
kill $API_PID || true
kill $HYPERVISOR_PID || true

print_status "Test deployment completed!"
exit 0
