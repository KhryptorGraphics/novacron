# Cloud Provider Examples

This directory contains examples for working with the cloud provider abstraction layer in Novacron.

## Directory Structure

- `basic/` - Basic examples of working with cloud providers
- `testing/` - Examples of testing cloud provider implementations
- `testsuite/` - A reusable test suite for cloud providers

## Basic Example

The basic example in `basic/` demonstrates:

- Initializing cloud providers (AWS, Azure, GCP)
- Listing instances across all providers
- Creating instances
- Finding the cheapest provider for a specific instance type
- Comparing pricing across providers
- Using the hybrid cloud orchestrator

To run the basic example:

```bash
cd basic
go run main.go
```

## Testing Example

The testing example in `testing/` demonstrates:

- Setting up a test suite for cloud providers
- Running tests for specific provider operations
- Running a multi-provider test scenario
- Testing cloud provider operations (instances, storage, networks)

To run the testing example:

```bash
cd testing
go run main.go                   # Run a multi-provider test scenario
go run main.go all               # Run all tests for all providers
go run main.go aws               # Run tests for AWS provider
go run main.go azure             # Run tests for Azure provider
go run main.go gcp               # Run tests for GCP provider
```

## Test Suite

The `testsuite/` directory contains a reusable test suite for cloud providers. This test suite can be used to validate cloud provider implementations against the standard cloud provider interface.

The test suite includes tests for:

- Instance operations
- Image operations
- Region operations
- Storage operations
- Snapshot operations
- Network operations
- Pricing operations

To use the test suite in your own code:

```go
import "github.com/novacron/backend/examples/cloud/testsuite"

// Create a new test suite
testSuite := testsuite.NewProviderTestSuite(providerManager, true)

// Run all tests for all providers
testSuite.RunAllTests(ctx)

// Or run specific tests
testSuite.TestInstanceOperations(ctx, "aws", awsProvider)
```

## Environment Variables

The examples use the following environment variables for cloud provider authentication:

### AWS
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN`

### Azure
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `AZURE_SUBSCRIPTION_ID`

### GCP
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GCP_PROJECT_ID`

If these environment variables are not provided, the examples will use dummy values for testing purposes.
