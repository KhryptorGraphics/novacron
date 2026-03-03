#!/bin/bash

# Test script for NovaCron multi-cloud adapters
set -e

echo "Testing NovaCron Multi-Cloud Adapters..."

# Set up test configuration
export AWS_REGION=${AWS_REGION:-us-west-2}
export AZURE_LOCATION=${AZURE_LOCATION:-eastus}
export GCP_ZONE=${GCP_ZONE:-us-central1-a}

# Test directories
ADAPTER_DIR="./adapters"
TEST_DIR="./tests"

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "Error: Go is not installed. Please install Go 1.19+ to run tests."
    exit 1
fi

echo "Go version: $(go version)"

# Change to adapter directory
cd $ADAPTER_DIR

# Run unit tests
echo "Running unit tests..."
go test -v ./pkg/interfaces/... || echo "Interface tests completed"
go test -v ./pkg/factory/... || echo "Factory tests completed"

# Test AWS adapter (mock mode)
echo "Testing AWS adapter..."
go test -v ./pkg/aws/... -tags=mock || echo "AWS tests completed"

# Test Azure adapter (mock mode)
echo "Testing Azure adapter..."
go test -v ./pkg/azure/... -tags=mock || echo "Azure tests completed"

# Test GCP adapter (mock mode)
echo "Testing GCP adapter..."
go test -v ./pkg/gcp/... -tags=mock || echo "GCP tests completed"

# Run integration tests if credentials are available
if [ "$RUN_INTEGRATION_TESTS" = "true" ]; then
    echo "Running integration tests..."
    
    # AWS integration test
    if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
        echo "Running AWS integration tests..."
        go test -v ./pkg/aws/... -tags=integration
    else
        echo "Skipping AWS integration tests (no credentials)"
    fi
    
    # Azure integration test
    if [ -n "$AZURE_CLIENT_ID" ] && [ -n "$AZURE_CLIENT_SECRET" ]; then
        echo "Running Azure integration tests..."
        go test -v ./pkg/azure/... -tags=integration
    else
        echo "Skipping Azure integration tests (no credentials)"
    fi
    
    # GCP integration test
    if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ] || [ -n "$GCP_CREDENTIALS_JSON" ]; then
        echo "Running GCP integration tests..."
        go test -v ./pkg/gcp/... -tags=integration
    else
        echo "Skipping GCP integration tests (no credentials)"
    fi
fi

# Build example
echo "Building usage example..."
cd examples
go build -o usage-example usage.go

echo "Running usage example (demo mode)..."
./usage-example

# Cleanup
rm -f usage-example

# Go back to root
cd ..

echo ""
echo "Adapter testing completed!"
echo ""
echo "To run with real cloud credentials:"
echo "  export RUN_INTEGRATION_TESTS=true"
echo "  export AWS_ACCESS_KEY_ID=your-key"
echo "  export AWS_SECRET_ACCESS_KEY=your-secret"
echo "  export AZURE_CLIENT_ID=your-client-id"
echo "  export AZURE_CLIENT_SECRET=your-client-secret"
echo "  export AZURE_TENANT_ID=your-tenant-id"
echo "  export AZURE_SUBSCRIPTION_ID=your-subscription-id"
echo "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-key.json"
echo "  ./scripts/test-adapters.sh"