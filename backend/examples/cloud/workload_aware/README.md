# Workload-Aware Cloud Resource Management Example

This example demonstrates the enhanced hybrid cloud orchestrator's workload-aware capabilities. It showcases how different workload types are automatically matched with the most appropriate cloud provider and instance type based on their characteristics.

## Overview

The workload-aware orchestrator:

1. Analyzes workload characteristics (CPU, memory, IO patterns, etc.)
2. Classifies the workload into specific types (web server, database, ML training, etc.)
3. Determines the best provider and instance type for each workload
4. Makes placement decisions that optimize for cost, performance, or compliance

## Workload Types Supported

- Web servers
- Batch processing jobs
- Database workloads
- Machine learning training
- Machine learning inference
- Analytics workloads
- Development and testing environments

## Running the Example

```bash
# Run the example with all workload types
go run main.go

# Run the example with a specific workload type
go run main.go web
go run main.go batch
go run main.go database
go run main.go ml-training
go run main.go ml-inference
go run main.go analytics
go run main.go dev-test

# Show help
go run main.go help
```

## How It Works

1. The example sets up a hybrid cloud orchestrator with AWS, Azure, and GCP providers
2. It creates different workload profiles with realistic metrics for each workload type
3. For each workload, it determines the best cloud provider and instance type
4. It creates an instance for each workload using the selected provider
5. Finally, it generates cost optimization recommendations based on the workload profiles

## Key Features Demonstrated

- **Workload Classification**: Automatically identifies workload type based on metrics
- **Provider Selection**: Chooses the best provider for each workload type
- **Instance Type Optimization**: Selects the optimal instance type for each workload
- **Cost Optimization**: Generates recommendations for cost savings
- **Multi-Cloud Management**: Works across multiple cloud providers

## Integration with The Project

This example is part of the Novacron project's hybrid cloud capabilities:

- Workload-aware placement
- Multi-cloud orchestration
- Cost optimization
- Resource efficiency

The enhanced hybrid cloud orchestrator improves resource utilization and reduces costs by matching workloads to the most appropriate cloud resources based on their characteristics.
