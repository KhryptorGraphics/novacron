# NovaCron - Distributed Cloud Hypervisor & Workflow Engine

NovaCron is a next-generation distributed cloud hypervisor and workflow engine that provides lightweight virtualization, automatic discovery, and advanced scheduling capabilities.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [API Documentation](#api-documentation)
4. [Frontend Dashboard](#frontend-dashboard)
5. [Deployment](#deployment)
6. [Monitoring & Observability](#monitoring--observability)
7. [Testing](#testing)
8. [Contributing](#contributing)

## Architecture Overview

NovaCron follows a microservices architecture with the following key components:

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Frontend      │    │    API Layer     │    │  Hypervisor      │
│   (Next.js)     │◄──►│   (Express.js)   │◄──►│  (Go/Process)    │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                              │                        │
                              ▼                        ▼
                    ┌──────────────────┐    ┌──────────────────┐
                    │   Database       │    │   VM Resources   │
                    │   (MySQL)        │    │   (Host System)  │
                    └──────────────────┘    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Redis          │
                    │   (Scheduling)   │
                    └──────────────────┘
```

## Core Components

### 1. Hypervisor (Go)

The hypervisor provides lightweight virtualization using Linux namespaces and cgroups. Key features include:

- Process-based isolation
- Automatic resource management
- Live migration capabilities
- Cross-cluster federation

### 2. Scheduler & Workflow Engine (TypeScript)

The scheduler handles job execution and workflow orchestration:

- Cron-based job scheduling
- Distributed locking with Redis
- Workflow DAG execution
- Retry mechanisms with exponential backoff

### 3. Database Layer (MySQL)

Persistent storage for jobs, workflows, and execution history:

- Job definitions and scheduling information
- Workflow definitions and execution state
- Execution metrics and logs

### 4. Frontend Dashboard (Next.js)

Web-based management interface:

- Real-time monitoring
- Job and workflow management
- Execution visualization
- System status overview

## API Documentation

### Jobs API

#### Create a Job
```http
POST /api/jobs
Content-Type: application/json

{
  "name": "Daily Backup",
  "schedule": "0 2 * * *",
  "timezone": "UTC",
  "enabled": true,
  "priority": 5,
  "max_retries": 3,
  "timeout": 30000
}
```

#### List Jobs
```http
GET /api/jobs
```

#### Get Job Details
```http
GET /api/jobs/{id}
```

#### Update Job
```http
PUT /api/jobs/{id}
Content-Type: application/json

{
  "name": "Updated Job Name",
  "enabled": false
}
```

#### Delete Job
```http
DELETE /api/jobs/{id}
```

#### Execute Job Immediately
```http
POST /api/jobs/{id}/execute
```

### Workflows API

#### Create a Workflow
```http
POST /api/workflows
Content-Type: application/json

{
  "name": "Data Processing Pipeline",
  "description": "Process and analyze daily data",
  "nodes": [
    {
      "id": "1",
      "name": "Data Ingestion",
      "type": "job",
      "config": {
        "jobId": "ingest-data-job"
      }
    },
    {
      "id": "2",
      "name": "Data Processing",
      "type": "job",
      "config": {
        "jobId": "process-data-job"
      }
    }
  ],
  "edges": [
    {
      "from": "1",
      "to": "2"
    }
  ],
  "enabled": true
}
```

#### List Workflows
```http
GET /api/workflows
```

#### Get Workflow Details
```http
GET /api/workflows/{id}
```

#### Execute Workflow
```http
POST /api/workflows/{id}/execute
```

## Frontend Dashboard

The NovaCron dashboard provides a comprehensive interface for managing the system:

### Overview Tab
- System health status
- Execution metrics and statistics
- Recent activity timeline

### Jobs Tab
- Job creation and management
- Schedule configuration
- Execution history and monitoring

### Workflows Tab
- Workflow creation and editing
- Visual workflow designer
- Execution tracking

### Visualization Tab
- Interactive workflow graphs
- Real-time execution status
- Node details and configuration

### Monitoring Tab
- Detailed execution metrics
- Performance analytics
- Error tracking and debugging

## Deployment

### Docker Compose (Development)
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Docker Compose (Production)
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | Database host | localhost |
| `DB_PORT` | Database port | 3306 |
| `DB_USER` | Database user | root |
| `DB_PASSWORD` | Database password | password |
| `DB_NAME` | Database name | novacron |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379 |
| `PORT` | API server port | 8090 |

## Monitoring & Observability

NovaCron includes built-in monitoring capabilities:

### Prometheus Metrics
- Job execution duration
- Success/failure rates
- System resource usage
- API response times

### Grafana Dashboards
Pre-configured dashboards for:
- Job execution overview
- Workflow performance
- System health
- Resource utilization

### Logging
Structured logging with:
- Request tracing
- Error reporting
- Performance metrics
- Audit trails

## Testing

### Unit Tests
```bash
npm run test:unit
```

### Integration Tests
```bash
npm run test:integration
```

### End-to-End Tests
```bash
cd frontend && npm run test:e2e
```

### Test Coverage
- API layer testing
- Component testing
- Database integration testing
- Workflow execution testing

## Contributing

### Development Setup
1. Clone the repository
2. Install dependencies: `npm install`
3. Start development services: `docker-compose -f docker-compose.dev.yml up -d`
4. Run database migrations: `npm run migrate`
5. Start frontend: `cd frontend && npm run dev`
6. Start backend: `npm run dev`

### Code Standards
- Follow TypeScript/JavaScript best practices
- Use ESLint and Prettier for code formatting
- Write comprehensive tests for new features
- Document public APIs and components

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation as needed
5. Submit pull request with description

### Release Process
1. Update version in package.json
2. Create release branch
3. Run full test suite
4. Create Git tag
5. Publish to Docker Hub
6. Update documentation