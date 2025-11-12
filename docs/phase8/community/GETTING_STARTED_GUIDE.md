# DWCP v3 Getting Started Guide

**Zero to Production in 30 Minutes**

Version: 3.0.0
Last Updated: 2025-11-10
Target Audience: Developers, DevOps Engineers, System Architects

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Installation Methods](#installation-methods)
5. [First Application](#first-application)
6. [Configuration](#configuration)
7. [Development Workflow](#development-workflow)
8. [Testing](#testing)
9. [Deployment](#deployment)
10. [Monitoring](#monitoring)
11. [Troubleshooting](#troubleshooting)
12. [Next Steps](#next-steps)

---

## Introduction

### What is DWCP v3?

The Distributed Workspace Communication Protocol (DWCP) v3 is a next-generation distributed computing platform that provides:

- **Extreme Scale**: Support for 1M+ nodes with linear scalability
- **Byzantine Fault Tolerance**: Secure operation with up to 33% malicious nodes
- **Multi-Protocol Support**: gRPC, WebSocket, HTTP/3, QUIC
- **Zero-Trust Security**: End-to-end encryption with quantum resistance
- **Neural Optimization**: ML-driven performance tuning
- **Cloud-Native**: Kubernetes-ready with GitOps support

### Use Cases

- **Distributed AI Training**: Coordinate 10,000+ GPU nodes
- **Edge Computing**: Manage millions of IoT devices
- **Financial Systems**: High-frequency trading infrastructure
- **Healthcare**: HIPAA-compliant distributed systems
- **Gaming**: Massive multiplayer online games
- **Supply Chain**: Global logistics coordination

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DWCP v3 Platform                     │
├─────────────────────────────────────────────────────────┤
│  Application Layer                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ REST API │ │ GraphQL  │ │WebSocket │ │  gRPC    │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
├─────────────────────────────────────────────────────────┤
│  Consensus Layer                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │   Raft   │ │ Byzantine│ │  Gossip  │              │
│  └──────────┘ └──────────┘ └──────────┘              │
├─────────────────────────────────────────────────────────┤
│  Transport Layer                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │  QUIC    │ │ HTTP/3   │ │WebSocket │ │  gRPC    │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
├─────────────────────────────────────────────────────────┤
│  Storage Layer                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │  Redis   │ │PostgreSQL│ │  S3/Minio│              │
│  └──────────┘ └──────────┘ └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

### Key Features

#### 1. Distributed Consensus
- **Raft**: Leader-based consensus for 3-7 nodes
- **Byzantine**: BFT consensus for 4-100 nodes
- **Gossip**: Eventually consistent for 100+ nodes

#### 2. Security
- TLS 1.3 with quantum-resistant algorithms
- mTLS for service-to-service communication
- Hardware security module (HSM) integration
- Zero-trust network architecture

#### 3. Performance
- Sub-millisecond latency for local clusters
- 100,000+ transactions per second per node
- Automatic load balancing and failover
- Neural network optimization

#### 4. Observability
- Distributed tracing (OpenTelemetry)
- Prometheus metrics
- Grafana dashboards
- Real-time health monitoring

---

## Prerequisites

### System Requirements

#### Minimum (Development)
- **CPU**: 4 cores (x86_64 or ARM64)
- **RAM**: 8 GB
- **Storage**: 20 GB SSD
- **Network**: 100 Mbps
- **OS**: Linux (Ubuntu 20.04+), macOS 12+, Windows 11 with WSL2

#### Recommended (Production)
- **CPU**: 16+ cores with AVX-512
- **RAM**: 64 GB ECC
- **Storage**: 500 GB NVMe SSD
- **Network**: 10 Gbps with RDMA
- **OS**: Ubuntu 22.04 LTS or RHEL 9

#### Enterprise (Large Scale)
- **CPU**: 64+ cores with SGX/SEV
- **RAM**: 256 GB+ ECC
- **Storage**: 2 TB NVMe in RAID 10
- **Network**: 100 Gbps with SR-IOV
- **OS**: Ubuntu 22.04 LTS with real-time kernel

### Software Dependencies

#### Required
```bash
# Node.js 18+ (LTS recommended)
node --version  # v20.11.0+

# Rust 1.75+ (for native modules)
rustc --version  # 1.75.0+

# Docker 24+ (for containerization)
docker --version  # 24.0.0+

# Kubernetes 1.28+ (for orchestration)
kubectl version  # 1.28.0+
```

#### Optional (Recommended)
```bash
# PostgreSQL 15+ (for metadata storage)
psql --version  # 15.0+

# Redis 7+ (for caching and pub/sub)
redis-cli --version  # 7.2.0+

# Prometheus 2.45+ (for metrics)
prometheus --version  # 2.45.0+

# Grafana 10+ (for visualization)
grafana-server --version  # 10.0.0+
```

### Network Requirements

#### Ports
- **50051**: gRPC API
- **8080**: HTTP/REST API
- **8081**: WebSocket API
- **9090**: Prometheus metrics
- **9091**: Health checks
- **7946**: Gossip protocol (TCP/UDP)
- **4789**: VXLAN overlay network

#### Firewall Rules
```bash
# Allow inbound DWCP traffic
sudo ufw allow 50051/tcp comment "DWCP gRPC"
sudo ufw allow 8080/tcp comment "DWCP HTTP"
sudo ufw allow 8081/tcp comment "DWCP WebSocket"
sudo ufw allow 9090/tcp comment "Prometheus metrics"
sudo ufw allow 7946/tcp comment "Gossip protocol"
sudo ufw allow 7946/udp comment "Gossip protocol"
```

### Cloud Provider Setup

#### AWS
```bash
# Create VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=dwcp-vpc}]'

# Create subnet
aws ec2 create-subnet --vpc-id vpc-xxx \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-1a

# Create security group
aws ec2 create-security-group \
  --group-name dwcp-sg \
  --description "DWCP security group" \
  --vpc-id vpc-xxx
```

#### GCP
```bash
# Create VPC network
gcloud compute networks create dwcp-network \
  --subnet-mode=custom

# Create subnet
gcloud compute networks subnets create dwcp-subnet \
  --network=dwcp-network \
  --region=us-central1 \
  --range=10.0.1.0/24

# Create firewall rules
gcloud compute firewall-rules create dwcp-allow-internal \
  --network=dwcp-network \
  --allow=tcp:50051,tcp:8080,tcp:8081
```

#### Azure
```bash
# Create resource group
az group create --name dwcp-rg --location eastus

# Create virtual network
az network vnet create \
  --resource-group dwcp-rg \
  --name dwcp-vnet \
  --address-prefix 10.0.0.0/16

# Create subnet
az network vnet subnet create \
  --resource-group dwcp-rg \
  --vnet-name dwcp-vnet \
  --name dwcp-subnet \
  --address-prefix 10.0.1.0/24
```

---

## Quick Start

### 5-Minute Setup

#### Step 1: Install DWCP CLI
```bash
# Using npm (recommended)
npm install -g @dwcp/cli@3.0.0

# Using yarn
yarn global add @dwcp/cli@3.0.0

# Using pnpm
pnpm add -g @dwcp/cli@3.0.0

# Verify installation
dwcp version
# Output: DWCP CLI v3.0.0
```

#### Step 2: Initialize Project
```bash
# Create new project
dwcp init my-dwcp-app

# Navigate to project
cd my-dwcp-app

# Project structure created:
# my-dwcp-app/
# ├── config/
# │   ├── development.yaml
# │   ├── production.yaml
# │   └── test.yaml
# ├── src/
# │   ├── index.ts
# │   └── services/
# ├── tests/
# ├── docker-compose.yml
# ├── Dockerfile
# └── package.json
```

#### Step 3: Start Development Server
```bash
# Start local cluster (3 nodes)
dwcp dev start

# Output:
# ✓ Starting DWCP development cluster...
# ✓ Node 1: http://localhost:8080
# ✓ Node 2: http://localhost:8081
# ✓ Node 3: http://localhost:8082
# ✓ Dashboard: http://localhost:9000
# ✓ Cluster ready in 12.3s
```

#### Step 4: Deploy First Service
```bash
# Generate service scaffold
dwcp generate service hello-world

# Service created at src/services/hello-world/

# Deploy service
dwcp deploy src/services/hello-world

# Output:
# ✓ Building service...
# ✓ Creating container image...
# ✓ Deploying to cluster...
# ✓ Service available at http://localhost:8080/hello-world
```

#### Step 5: Test Service
```bash
# Test HTTP endpoint
curl http://localhost:8080/hello-world

# Output:
# {
#   "message": "Hello from DWCP v3!",
#   "node": "dwcp-node-1",
#   "timestamp": "2025-11-10T22:54:55.000Z"
# }

# Test gRPC endpoint
grpcurl -plaintext localhost:50051 hello.HelloService/SayHello

# Output:
# {
#   "message": "Hello from DWCP v3!",
#   "node": "dwcp-node-1"
# }
```

### 30-Minute Production Setup

#### Complete Production Deployment

```bash
#!/bin/bash
# production-setup.sh

set -e

echo "🚀 DWCP v3 Production Setup"

# 1. Install dependencies
echo "📦 Installing dependencies..."
npm install -g @dwcp/cli@3.0.0
npm install -g @dwcp/tools@3.0.0

# 2. Create production project
echo "🏗️  Creating production project..."
dwcp init --template production my-prod-cluster
cd my-prod-cluster

# 3. Configure production settings
echo "⚙️  Configuring production..."
cat > config/production.yaml <<EOF
cluster:
  name: prod-cluster
  nodes: 7
  consensus: raft
  replication_factor: 3

security:
  tls:
    enabled: true
    cert_path: /etc/dwcp/certs/server.crt
    key_path: /etc/dwcp/certs/server.key
  authentication:
    type: mtls
    ca_cert: /etc/dwcp/certs/ca.crt

storage:
  type: postgresql
  connection: postgresql://dwcp:password@localhost:5432/dwcp
  pool_size: 100

monitoring:
  metrics:
    enabled: true
    port: 9090
  tracing:
    enabled: true
    endpoint: http://jaeger:14268/api/traces
  logging:
    level: info
    format: json

performance:
  max_connections: 10000
  max_requests_per_second: 100000
  timeout_seconds: 30
  enable_neural_optimization: true
EOF

# 4. Generate TLS certificates
echo "🔒 Generating TLS certificates..."
dwcp security generate-certs \
  --ca-name "DWCP Production CA" \
  --domains "*.dwcp.prod.example.com" \
  --output /etc/dwcp/certs

# 5. Initialize database
echo "💾 Initializing database..."
dwcp db init --migrate

# 6. Deploy cluster
echo "🌐 Deploying cluster..."
dwcp cluster deploy \
  --config config/production.yaml \
  --replicas 7 \
  --wait

# 7. Configure load balancer
echo "⚖️  Configuring load balancer..."
dwcp lb configure \
  --algorithm weighted-round-robin \
  --health-check-interval 5s

# 8. Enable monitoring
echo "📊 Enabling monitoring..."
dwcp monitoring enable \
  --prometheus \
  --grafana \
  --jaeger

# 9. Run health checks
echo "🏥 Running health checks..."
dwcp health check --all

# 10. Display cluster info
echo "✅ Production cluster ready!"
dwcp cluster info

echo ""
echo "🎉 Setup complete!"
echo "Dashboard: https://dashboard.dwcp.prod.example.com"
echo "API: https://api.dwcp.prod.example.com"
echo "Docs: https://docs.dwcp.prod.example.com"
```

---

## Installation Methods

### Method 1: NPM Package

#### Global Installation
```bash
# Install CLI globally
npm install -g @dwcp/cli@3.0.0

# Install SDK for application development
npm install @dwcp/sdk@3.0.0

# Install additional tools
npm install -g @dwcp/tools@3.0.0
```

#### Local Installation
```bash
# Initialize npm project
npm init -y

# Install DWCP packages
npm install @dwcp/cli @dwcp/sdk @dwcp/types

# Add scripts to package.json
cat >> package.json <<EOF
{
  "scripts": {
    "dev": "dwcp dev start",
    "build": "dwcp build",
    "test": "dwcp test",
    "deploy": "dwcp deploy"
  }
}
EOF
```

### Method 2: Docker

#### Using Docker Compose
```yaml
# docker-compose.yml
version: '3.9'

services:
  dwcp-node-1:
    image: dwcp/node:3.0.0
    container_name: dwcp-node-1
    environment:
      - NODE_ID=1
      - CLUSTER_NAME=dev-cluster
      - CONSENSUS_TYPE=raft
      - PEERS=dwcp-node-2:50051,dwcp-node-3:50051
    ports:
      - "8080:8080"
      - "50051:50051"
    volumes:
      - dwcp-data-1:/var/lib/dwcp
      - ./config:/etc/dwcp
    networks:
      - dwcp-network

  dwcp-node-2:
    image: dwcp/node:3.0.0
    container_name: dwcp-node-2
    environment:
      - NODE_ID=2
      - CLUSTER_NAME=dev-cluster
      - CONSENSUS_TYPE=raft
      - PEERS=dwcp-node-1:50051,dwcp-node-3:50051
    ports:
      - "8081:8080"
      - "50052:50051"
    volumes:
      - dwcp-data-2:/var/lib/dwcp
      - ./config:/etc/dwcp
    networks:
      - dwcp-network

  dwcp-node-3:
    image: dwcp/node:3.0.0
    container_name: dwcp-node-3
    environment:
      - NODE_ID=3
      - CLUSTER_NAME=dev-cluster
      - CONSENSUS_TYPE=raft
      - PEERS=dwcp-node-1:50051,dwcp-node-2:50051
    ports:
      - "8082:8080"
      - "50053:50051"
    volumes:
      - dwcp-data-3:/var/lib/dwcp
      - ./config:/etc/dwcp
    networks:
      - dwcp-network

  postgres:
    image: postgres:15
    container_name: dwcp-postgres
    environment:
      - POSTGRES_DB=dwcp
      - POSTGRES_USER=dwcp
      - POSTGRES_PASSWORD=dwcp_password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - dwcp-network

  redis:
    image: redis:7
    container_name: dwcp-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - dwcp-network

  prometheus:
    image: prom/prometheus:latest
    container_name: dwcp-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - dwcp-network

  grafana:
    image: grafana/grafana:latest
    container_name: dwcp-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning
    networks:
      - dwcp-network

volumes:
  dwcp-data-1:
  dwcp-data-2:
  dwcp-data-3:
  postgres-data:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  dwcp-network:
    driver: bridge
```

#### Start Cluster
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f dwcp-node-1

# Stop cluster
docker-compose down
```

### Method 3: Kubernetes

#### Using Helm Chart
```bash
# Add DWCP Helm repository
helm repo add dwcp https://charts.dwcp.io
helm repo update

# Install DWCP cluster
helm install my-cluster dwcp/dwcp \
  --namespace dwcp \
  --create-namespace \
  --set cluster.nodes=7 \
  --set consensus.type=raft \
  --set security.tls.enabled=true

# Check deployment status
kubectl get pods -n dwcp

# Access dashboard
kubectl port-forward -n dwcp svc/dwcp-dashboard 9000:9000
```

#### Manual Kubernetes Deployment
```yaml
# dwcp-deployment.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: dwcp

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: dwcp-config
  namespace: dwcp
data:
  config.yaml: |
    cluster:
      name: k8s-cluster
      consensus: raft
    security:
      tls:
        enabled: true

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: dwcp-node
  namespace: dwcp
spec:
  serviceName: dwcp
  replicas: 7
  selector:
    matchLabels:
      app: dwcp
  template:
    metadata:
      labels:
        app: dwcp
    spec:
      containers:
      - name: dwcp
        image: dwcp/node:3.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 50051
          name: grpc
        env:
        - name: NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        volumeMounts:
        - name: data
          mountPath: /var/lib/dwcp
        - name: config
          mountPath: /etc/dwcp
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: dwcp
  namespace: dwcp
spec:
  clusterIP: None
  selector:
    app: dwcp
  ports:
  - name: http
    port: 8080
  - name: grpc
    port: 50051
```

### Method 4: Binary Installation

#### Download Pre-built Binaries
```bash
# Linux (x86_64)
curl -LO https://github.com/dwcp/dwcp/releases/download/v3.0.0/dwcp-linux-amd64.tar.gz
tar xzf dwcp-linux-amd64.tar.gz
sudo mv dwcp /usr/local/bin/

# macOS (ARM64)
curl -LO https://github.com/dwcp/dwcp/releases/download/v3.0.0/dwcp-darwin-arm64.tar.gz
tar xzf dwcp-darwin-arm64.tar.gz
sudo mv dwcp /usr/local/bin/

# Windows
curl -LO https://github.com/dwcp/dwcp/releases/download/v3.0.0/dwcp-windows-amd64.zip
unzip dwcp-windows-amd64.zip
# Add to PATH

# Verify installation
dwcp --version
```

### Method 5: Build from Source

#### Prerequisites
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install build tools
sudo apt-get install -y build-essential pkg-config libssl-dev
```

#### Build Steps
```bash
# Clone repository
git clone https://github.com/dwcp/dwcp.git
cd dwcp

# Install dependencies
npm install

# Build native modules
cd native
cargo build --release
cd ..

# Build TypeScript
npm run build

# Run tests
npm test

# Create distribution
npm run package

# Install globally
npm link
```

---

## First Application

### Hello World Service

#### Step 1: Create Service
```typescript
// src/services/hello-world/index.ts

import { DWCPService, Context, Request, Response } from '@dwcp/sdk';

export class HelloWorldService extends DWCPService {
  constructor() {
    super({
      name: 'hello-world',
      version: '1.0.0',
      endpoints: [
        { path: '/hello', method: 'GET', handler: this.sayHello },
        { path: '/hello/:name', method: 'GET', handler: this.sayHelloTo }
      ]
    });
  }

  async sayHello(ctx: Context, req: Request): Promise<Response> {
    return {
      status: 200,
      body: {
        message: 'Hello from DWCP v3!',
        node: ctx.node.id,
        timestamp: new Date().toISOString()
      }
    };
  }

  async sayHelloTo(ctx: Context, req: Request): Promise<Response> {
    const { name } = req.params;

    return {
      status: 200,
      body: {
        message: `Hello, ${name}!`,
        node: ctx.node.id,
        timestamp: new Date().toISOString()
      }
    };
  }
}

// Register service
export default new HelloWorldService();
```

#### Step 2: Configure Service
```yaml
# src/services/hello-world/config.yaml

service:
  name: hello-world
  version: 1.0.0
  replicas: 3

resources:
  cpu: 0.5
  memory: 512Mi

scaling:
  min_replicas: 1
  max_replicas: 10
  target_cpu_utilization: 70

health_check:
  path: /health
  interval: 10s
  timeout: 5s

monitoring:
  metrics: true
  tracing: true
```

#### Step 3: Deploy Service
```bash
# Deploy to development cluster
dwcp deploy src/services/hello-world

# Output:
# ✓ Building service hello-world...
# ✓ Creating container image...
# ✓ Pushing to registry...
# ✓ Deploying to cluster...
# ✓ Service deployed successfully
#
# Endpoints:
#   HTTP: http://localhost:8080/hello
#   gRPC: localhost:50051/hello.HelloService
#
# Status:
#   Replicas: 3/3
#   Health: 100%
```

#### Step 4: Test Service
```bash
# Test HTTP endpoint
curl http://localhost:8080/hello

# Test with parameter
curl http://localhost:8080/hello/Alice

# Test gRPC
grpcurl -plaintext localhost:50051 hello.HelloService/SayHello

# Load test
ab -n 10000 -c 100 http://localhost:8080/hello
```

### Distributed Counter Service

#### Advanced Example with State Management

```typescript
// src/services/counter/index.ts

import { DWCPService, Context, Request, Response, StateManager } from '@dwcp/sdk';

interface CounterState {
  value: number;
  updates: number;
  lastUpdate: string;
}

export class CounterService extends DWCPService {
  private state: StateManager<CounterState>;

  constructor() {
    super({
      name: 'counter',
      version: '1.0.0',
      stateful: true,
      consensus: 'raft'
    });

    this.state = new StateManager({
      initial: { value: 0, updates: 0, lastUpdate: new Date().toISOString() },
      replication: 3,
      consistency: 'strong'
    });
  }

  async increment(ctx: Context, req: Request): Promise<Response> {
    const delta = req.body?.delta || 1;

    // Acquire distributed lock
    const lock = await ctx.cluster.lock('counter', { timeout: 5000 });

    try {
      // Read current state
      const current = await this.state.get();

      // Update state
      const updated: CounterState = {
        value: current.value + delta,
        updates: current.updates + 1,
        lastUpdate: new Date().toISOString()
      };

      // Write with consensus
      await this.state.set(updated);

      // Broadcast event
      await ctx.cluster.broadcast('counter.incremented', {
        delta,
        newValue: updated.value
      });

      return {
        status: 200,
        body: updated
      };
    } finally {
      // Release lock
      await lock.release();
    }
  }

  async get(ctx: Context, req: Request): Promise<Response> {
    const state = await this.state.get();

    return {
      status: 200,
      body: state
    };
  }

  async reset(ctx: Context, req: Request): Promise<Response> {
    const lock = await ctx.cluster.lock('counter', { timeout: 5000 });

    try {
      await this.state.set({
        value: 0,
        updates: 0,
        lastUpdate: new Date().toISOString()
      });

      return {
        status: 200,
        body: { message: 'Counter reset' }
      };
    } finally {
      await lock.release();
    }
  }
}

export default new CounterService();
```

### Real-time Chat Service

#### WebSocket-based Distributed Chat

```typescript
// src/services/chat/index.ts

import { DWCPService, Context, WebSocketConnection, Message } from '@dwcp/sdk';

interface ChatMessage {
  id: string;
  user: string;
  text: string;
  timestamp: string;
  node: string;
}

export class ChatService extends DWCPService {
  private connections: Map<string, WebSocketConnection> = new Map();
  private messageHistory: ChatMessage[] = [];

  constructor() {
    super({
      name: 'chat',
      version: '1.0.0',
      websocket: true
    });

    // Subscribe to cluster events
    this.onClusterEvent('chat.message', this.handleClusterMessage);
  }

  async onConnect(ctx: Context, conn: WebSocketConnection): Promise<void> {
    const userId = conn.query.userId || `user-${Date.now()}`;

    this.connections.set(userId, conn);

    // Send message history
    await conn.send({
      type: 'history',
      messages: this.messageHistory.slice(-100)
    });

    // Broadcast join event
    await ctx.cluster.broadcast('chat.user.joined', {
      userId,
      timestamp: new Date().toISOString()
    });
  }

  async onMessage(ctx: Context, conn: WebSocketConnection, msg: Message): Promise<void> {
    const userId = conn.query.userId;

    if (msg.type === 'message') {
      const chatMessage: ChatMessage = {
        id: `msg-${Date.now()}`,
        user: userId,
        text: msg.text,
        timestamp: new Date().toISOString(),
        node: ctx.node.id
      };

      // Store locally
      this.messageHistory.push(chatMessage);

      // Broadcast to all nodes
      await ctx.cluster.broadcast('chat.message', chatMessage);

      // Send to local connections
      await this.broadcastToLocal(chatMessage);
    }
  }

  async onDisconnect(ctx: Context, conn: WebSocketConnection): Promise<void> {
    const userId = conn.query.userId;
    this.connections.delete(userId);

    // Broadcast leave event
    await ctx.cluster.broadcast('chat.user.left', {
      userId,
      timestamp: new Date().toISOString()
    });
  }

  private async handleClusterMessage(event: any): Promise<void> {
    await this.broadcastToLocal(event.data);
  }

  private async broadcastToLocal(message: ChatMessage): Promise<void> {
    for (const conn of this.connections.values()) {
      await conn.send({
        type: 'message',
        ...message
      });
    }
  }
}

export default new ChatService();
```

---

## Configuration

### Configuration Files

#### Development Configuration
```yaml
# config/development.yaml

cluster:
  name: dev-cluster
  nodes: 3
  consensus: raft

networking:
  http:
    port: 8080
    cors:
      enabled: true
      origins: ['*']
  grpc:
    port: 50051
  websocket:
    port: 8081

security:
  tls:
    enabled: false
  authentication:
    type: none

storage:
  type: memory
  persistence: false

logging:
  level: debug
  format: pretty

monitoring:
  metrics:
    enabled: true
  tracing:
    enabled: false
```

#### Production Configuration
```yaml
# config/production.yaml

cluster:
  name: prod-cluster
  nodes: 7
  consensus: raft
  replication_factor: 3
  election_timeout: 5000ms
  heartbeat_interval: 1000ms

networking:
  http:
    port: 8080
    host: 0.0.0.0
    cors:
      enabled: true
      origins:
        - https://app.example.com
        - https://dashboard.example.com
    rate_limit:
      enabled: true
      requests_per_second: 1000
      burst: 100

  grpc:
    port: 50051
    max_connections: 10000
    keepalive_interval: 30s

  websocket:
    port: 8081
    max_connections: 100000
    ping_interval: 30s

security:
  tls:
    enabled: true
    cert_path: /etc/dwcp/certs/server.crt
    key_path: /etc/dwcp/certs/server.key
    ca_cert: /etc/dwcp/certs/ca.crt
    min_version: TLS1.3
    cipher_suites:
      - TLS_AES_256_GCM_SHA384
      - TLS_CHACHA20_POLY1305_SHA256

  authentication:
    type: mtls
    require_client_cert: true

  authorization:
    enabled: true
    type: rbac
    policy_file: /etc/dwcp/policies/rbac.yaml

storage:
  type: postgresql
  connection: postgresql://dwcp:${DB_PASSWORD}@postgres:5432/dwcp
  pool_size: 100
  max_connections: 1000

  cache:
    enabled: true
    type: redis
    connection: redis://redis:6379
    ttl: 3600s

  backup:
    enabled: true
    schedule: "0 2 * * *"
    retention_days: 30
    storage: s3://dwcp-backups/

logging:
  level: info
  format: json
  outputs:
    - type: stdout
    - type: file
      path: /var/log/dwcp/dwcp.log
      max_size: 100MB
      max_age: 30
      max_backups: 10
    - type: syslog
      network: tcp
      address: syslog.example.com:514

monitoring:
  metrics:
    enabled: true
    port: 9090
    path: /metrics
    interval: 15s

  tracing:
    enabled: true
    type: opentelemetry
    endpoint: http://jaeger:14268/api/traces
    sample_rate: 0.1

  health:
    enabled: true
    port: 9091
    path: /health

  alerts:
    enabled: true
    webhook: https://alerts.example.com/webhook

performance:
  max_connections: 10000
  max_requests_per_second: 100000
  timeout:
    read: 30s
    write: 30s
    idle: 120s

  optimization:
    enable_neural: true
    enable_caching: true
    enable_compression: true
    compression_level: 6

  resource_limits:
    cpu: 16
    memory: 64GB
    disk: 500GB
```

### Environment Variables

```bash
# .env.production

# Cluster Configuration
DWCP_CLUSTER_NAME=prod-cluster
DWCP_NODE_ID=1
DWCP_CONSENSUS_TYPE=raft

# Networking
DWCP_HTTP_PORT=8080
DWCP_GRPC_PORT=50051
DWCP_WS_PORT=8081

# Security
DWCP_TLS_ENABLED=true
DWCP_TLS_CERT=/etc/dwcp/certs/server.crt
DWCP_TLS_KEY=/etc/dwcp/certs/server.key
DWCP_TLS_CA=/etc/dwcp/certs/ca.crt

# Database
DATABASE_URL=postgresql://dwcp:password@postgres:5432/dwcp
REDIS_URL=redis://redis:6379

# Monitoring
PROMETHEUS_PORT=9090
JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# Performance
MAX_CONNECTIONS=10000
MAX_RPS=100000

# AWS Integration (optional)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
S3_BUCKET=dwcp-backups
```

---

## Development Workflow

### Local Development

#### Start Development Environment
```bash
# Start local cluster with hot reload
dwcp dev start --watch

# Start with specific number of nodes
dwcp dev start --nodes 5

# Start with custom config
dwcp dev start --config config/custom.yaml
```

#### Development Commands
```bash
# Generate new service
dwcp generate service my-service

# Generate new middleware
dwcp generate middleware auth

# Generate new model
dwcp generate model User

# Run linter
dwcp lint

# Format code
dwcp format

# Type check
dwcp typecheck
```

### Testing Workflow

```bash
# Run all tests
dwcp test

# Run unit tests
dwcp test:unit

# Run integration tests
dwcp test:integration

# Run e2e tests
dwcp test:e2e

# Run with coverage
dwcp test --coverage

# Watch mode
dwcp test --watch
```

### Debugging

```bash
# Start cluster in debug mode
dwcp dev start --debug

# Attach debugger to specific node
dwcp debug attach dwcp-node-1

# View logs
dwcp logs --follow

# Inspect cluster state
dwcp cluster inspect

# Profile performance
dwcp profile start
```

---

## Testing

### Unit Testing

```typescript
// tests/services/hello-world.test.ts

import { describe, it, expect, beforeAll, afterAll } from '@dwcp/testing';
import { TestCluster } from '@dwcp/testing';
import HelloWorldService from '../src/services/hello-world';

describe('HelloWorldService', () => {
  let cluster: TestCluster;

  beforeAll(async () => {
    cluster = await TestCluster.create({
      nodes: 3,
      services: [HelloWorldService]
    });
  });

  afterAll(async () => {
    await cluster.destroy();
  });

  it('should return hello message', async () => {
    const response = await cluster.request('/hello');

    expect(response.status).toBe(200);
    expect(response.body.message).toBe('Hello from DWCP v3!');
    expect(response.body.node).toBeDefined();
  });

  it('should return personalized hello', async () => {
    const response = await cluster.request('/hello/Alice');

    expect(response.status).toBe(200);
    expect(response.body.message).toBe('Hello, Alice!');
  });

  it('should handle concurrent requests', async () => {
    const requests = Array(100).fill(null).map(() =>
      cluster.request('/hello')
    );

    const responses = await Promise.all(requests);

    expect(responses.every(r => r.status === 200)).toBe(true);
  });
});
```

### Integration Testing

```typescript
// tests/integration/cluster.test.ts

import { describe, it, expect } from '@dwcp/testing';
import { TestCluster } from '@dwcp/testing';

describe('Cluster Integration', () => {
  it('should maintain consensus during node failure', async () => {
    const cluster = await TestCluster.create({ nodes: 5 });

    // Write data
    await cluster.request('/counter/increment', {
      method: 'POST',
      body: { delta: 10 }
    });

    // Kill a node
    await cluster.killNode(2);

    // Verify data is still accessible
    const response = await cluster.request('/counter');
    expect(response.body.value).toBe(10);

    // Verify cluster is still functional
    await cluster.request('/counter/increment', {
      method: 'POST',
      body: { delta: 5 }
    });

    const final = await cluster.request('/counter');
    expect(final.body.value).toBe(15);

    await cluster.destroy();
  });

  it('should replicate state across nodes', async () => {
    const cluster = await TestCluster.create({ nodes: 3 });

    // Write to node 1
    await cluster.request('/counter/increment', {
      node: 1,
      method: 'POST',
      body: { delta: 100 }
    });

    // Read from different nodes
    const [r1, r2, r3] = await Promise.all([
      cluster.request('/counter', { node: 1 }),
      cluster.request('/counter', { node: 2 }),
      cluster.request('/counter', { node: 3 })
    ]);

    expect(r1.body.value).toBe(100);
    expect(r2.body.value).toBe(100);
    expect(r3.body.value).toBe(100);

    await cluster.destroy();
  });
});
```

### Load Testing

```bash
# HTTP load test
dwcp load test \
  --url http://localhost:8080/hello \
  --requests 100000 \
  --concurrency 1000 \
  --duration 60s

# gRPC load test
dwcp load test \
  --protocol grpc \
  --endpoint localhost:50051 \
  --service hello.HelloService \
  --method SayHello \
  --requests 100000 \
  --concurrency 1000

# WebSocket load test
dwcp load test \
  --protocol ws \
  --url ws://localhost:8081/chat \
  --connections 10000 \
  --messages-per-connection 100
```

---

## Deployment

### Development Deployment

```bash
# Deploy to local development cluster
dwcp deploy --env development

# Deploy specific service
dwcp deploy src/services/hello-world --env development

# Deploy with overrides
dwcp deploy --env development --replicas 5 --resources "cpu=2,memory=4Gi"
```

### Staging Deployment

```bash
# Deploy to staging
dwcp deploy --env staging

# Run smoke tests
dwcp test:smoke --env staging

# Promote to production if tests pass
if [ $? -eq 0 ]; then
  dwcp promote staging production
fi
```

### Production Deployment

#### Blue-Green Deployment
```bash
# Deploy new version to green environment
dwcp deploy --env production --strategy blue-green

# Verify green environment
dwcp health check --env green

# Switch traffic to green
dwcp traffic switch green

# Monitor for issues
dwcp monitor --env production --duration 1h

# Rollback if needed
dwcp rollback
```

#### Canary Deployment
```bash
# Deploy canary (10% traffic)
dwcp deploy --env production --strategy canary --traffic-split 10

# Monitor metrics
dwcp metrics compare production canary

# Increase traffic gradually
dwcp traffic shift --to canary --percentage 25
dwcp traffic shift --to canary --percentage 50
dwcp traffic shift --to canary --percentage 100

# Complete rollout
dwcp deploy finalize
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check rollout status
kubectl rollout status statefulset/dwcp-node -n dwcp

# Scale cluster
kubectl scale statefulset/dwcp-node --replicas=10 -n dwcp

# Update image
kubectl set image statefulset/dwcp-node dwcp=dwcp/node:3.0.1 -n dwcp

# Rollback
kubectl rollout undo statefulset/dwcp-node -n dwcp
```

---

## Monitoring

### Metrics

#### View Metrics
```bash
# View cluster metrics
dwcp metrics show

# Export metrics
dwcp metrics export --format prometheus

# Stream metrics
dwcp metrics stream --interval 5s
```

#### Key Metrics
- **Request Rate**: Requests per second
- **Error Rate**: Errors per second
- **Latency**: P50, P95, P99 response times
- **Throughput**: Bytes per second
- **CPU Usage**: Percentage per node
- **Memory Usage**: Bytes per node
- **Disk I/O**: Reads/writes per second
- **Network I/O**: Bytes in/out per second

### Dashboards

#### Grafana Dashboard
```bash
# Import DWCP dashboard
curl -o dwcp-dashboard.json https://grafana.com/api/dashboards/12345/revisions/1/download

grafana-cli dashboard import dwcp-dashboard.json
```

#### Dashboard URLs
- **Overview**: http://localhost:3000/d/dwcp-overview
- **Performance**: http://localhost:3000/d/dwcp-performance
- **Resources**: http://localhost:3000/d/dwcp-resources
- **Consensus**: http://localhost:3000/d/dwcp-consensus

### Alerts

```yaml
# config/alerts.yaml

groups:
  - name: dwcp_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(dwcp_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(dwcp_request_duration_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P99 latency is {{ $value }} seconds"

      - alert: NodeDown
        expr: up{job="dwcp"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "DWCP node is down"
          description: "Node {{ $labels.instance }} is unreachable"
```

---

## Troubleshooting

### Common Issues

#### Issue: Cluster fails to form
**Symptoms**: Nodes cannot reach each other

**Solution**:
```bash
# Check network connectivity
dwcp cluster diagnose network

# Check firewall rules
sudo ufw status

# Verify DNS resolution
nslookup dwcp-node-1

# Check node logs
dwcp logs dwcp-node-1
```

#### Issue: High memory usage
**Symptoms**: OOM errors, slow performance

**Solution**:
```bash
# Check memory usage
dwcp metrics show memory

# Analyze heap dump
dwcp debug heap-snapshot

# Increase memory limit
dwcp cluster update --memory 8Gi

# Enable memory profiling
dwcp profile memory
```

#### Issue: Slow consensus
**Symptoms**: High latency, timeouts

**Solution**:
```bash
# Check consensus metrics
dwcp metrics show consensus

# Increase timeouts
dwcp config set consensus.election_timeout 10000

# Optimize network
dwcp network tune

# Consider different consensus algorithm
dwcp cluster update --consensus gossip
```

### Debugging Tools

```bash
# Interactive debugging
dwcp debug shell

# Trace requests
dwcp trace start

# Dump cluster state
dwcp cluster dump --output state.json

# Validate configuration
dwcp config validate

# Run diagnostics
dwcp doctor
```

---

## Next Steps

### Learning Resources

1. **Architecture Deep Dive**: docs/ARCHITECTURE_DEEP_DIVE.md
2. **API Reference**: docs/api/
3. **Video Tutorials**: docs/tutorials/
4. **Best Practices**: docs/kb/best-practices.md
5. **Case Studies**: docs/kb/case-studies/

### Advanced Topics

- **Performance Tuning**: docs/PERFORMANCE_TUNING_GUIDE.md
- **Security Hardening**: docs/security/hardening.md
- **Multi-Region Deployment**: docs/deployment/multi-region.md
- **Neural Optimization**: docs/neural/optimization.md
- **Custom Consensus**: docs/advanced/custom-consensus.md

### Community

- **GitHub**: https://github.com/dwcp/dwcp
- **Discord**: https://discord.gg/dwcp
- **Forum**: https://forum.dwcp.io
- **Twitter**: @dwcp_io
- **Blog**: https://blog.dwcp.io

### Support

- **Documentation**: https://docs.dwcp.io
- **Enterprise Support**: support@dwcp.io
- **Training**: training@dwcp.io
- **Consulting**: consulting@dwcp.io

---

**Congratulations!** You've completed the DWCP v3 Getting Started Guide. You're now ready to build distributed applications at scale.

For questions or issues, please visit our [community forum](https://forum.dwcp.io) or [GitHub repository](https://github.com/dwcp/dwcp).

---

*Last updated: 2025-11-10*
*Version: 3.0.0*
*License: MIT*
