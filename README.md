# NovaCron

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

NovaCron is a distributed virtual machine management system with advanced VM migration capabilities, designed for reliability, performance, and extensibility.
If 50,000 people are running this software. Your running and os on a distributed processing platform consisting of 50,000 computers. 

## Features

### Core VM Migration Framework
- **Cold Migration**: Stops VM, transfers state, starts on new node
- **Warm Migration**: Suspends VM, transfers state, resumes on new node
- **Live Migration**: Uses iterative memory state transfer to minimize downtime
- **Automatic Rollback**: Reverts to original state if migration fails
- **Progress Tracking**: Real-time monitoring of migration status

### Hypervisor Management
- Supports multiple virtualization technologies (KVM, Containerd, Process)
- Resource optimization and allocation
- VM lifecycle management
- Real-time metrics and monitoring

### Architecture
- Microservices-based design
- RESTful API and WebSocket interfaces
- React-based dashboard with real-time updates
- Prometheus/Grafana integration for metrics

## Screenshots

<!-- TODO: Add screenshots here -->

## Quick Start

### Prerequisites
- Docker and Docker Compose
- For VM functionality: KVM/QEMU (Linux) or Hyper-V (Windows)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/novacron/novacron.git
cd novacron
```

2. Run the setup script:
```bash
# Linux/macOS
./scripts/setup.sh

# Windows (PowerShell)
.\scripts\setup.ps1
```

3. Open the web interface:
```
http://localhost:3000
```

## Components

### Backend

- **Core Engine (Go)**: VM lifecycle management, migration, and resource control
- **API Service (Go/Python)**: RESTful API for management operations
- **WebSocket Service (Python)**: Real-time events and updates

### Frontend

- **Web Dashboard (Next.js)**: Modern, responsive web interface
- **CLI (Go)**: Command-line interface for operations and scripting

### Infrastructure

- **Docker Containers**: Hypervisor, API, Frontend, Database
- **Prometheus/Grafana**: Monitoring and metrics visualization

## Architecture

```
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│               │  │               │  │               │
│  Dashboard    │  │  CLI Client   │  │  External     │
│               │  │               │  │  Integrations │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                 ┌─────────▼─────────┐
                 │                   │
                 │   API Gateway     │
                 │                   │
                 └─────────┬─────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────▼───────┐ ┌───────▼────────┐ ┌──────▼────────┐
│                │ │                │ │               │
│ Control Plane  │ │ VM Service     │ │ Auth Service  │
│                │ │                │ │               │
└────────┬───────┘ └───────┬────────┘ └───────────────┘
         │                 │
         │                 │
┌────────▼───────┐ ┌───────▼────────┐
│                │ │                │
│ Node Manager   │ │ Storage Service│
│                │ │                │
└────────┬───────┘ └────────────────┘
         │
         │
    ┌────▼────┐    ┌─────────┐    ┌─────────┐
    │         │    │         │    │         │
    │ Node 1  ├────► Node 2  ├────► Node N  │
    │         │    │         │    │         │
    └─────────┘    └─────────┘    └─────────┘
```

## Development

### Project Structure
```
novacron/
├── backend/               # Backend services
│   ├── core/              # Go core components
│   └── services/          # Python services
├── frontend/              # Next.js web interface
├── docker/                # Dockerfiles
├── configs/               # Configuration templates
├── scripts/               # Utility scripts
└── docs/                  # Documentation
```

### Building from Source

1. Build the backend:
```bash
cd backend/core
go build -o novacron-hypervisor ./cmd/novacron
```

2. Build the frontend:
```bash
cd frontend
npm install
npm run build
```

### Running Tests

```bash
# Go backend tests
cd backend/core
go test ./...

# Python service tests
cd backend/services
pytest

# Frontend tests
cd frontend
npm test
```

## VM Migration Types

NovaCron provides three types of VM migration strategies:

### Cold Migration
- VM is stopped on source node
- VM state is transferred to destination node
- VM is started on destination node
- High reliability, higher downtime

### Warm Migration
- VM is suspended (paused) on source node
- RAM and device state are transferred
- VM is resumed on destination node
- Medium downtime, good reliability

### Live Migration
- VM continues running during migration
- Memory pages are iteratively copied while VM runs
- Final brief pause for last memory pages and CPU state
- Minimal downtime, more complex

## Configuration

Configuration is handled through YAML files or environment variables:

### Sample Config
```yaml
nodeId: node1
logLevel: info
api:
  host: 0.0.0.0
  port: 8080
  tlsEnabled: false
```

### Environment Variables
- `NODE_ID`: Unique identifier for the node
- `LOG_LEVEL`: Logging level (debug, info, warn, error)
- `API_PORT`: Port for the API service
- `DB_URL`: Database connection string

## Contributing

Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for contribution guidelines.

## Roadmap

See [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for future plans and development roadmap.

## License

NovaCron is released under the MIT License. See [LICENSE](LICENSE) for details.
