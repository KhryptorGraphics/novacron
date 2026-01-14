# NovaCron Jetson Thor Setup Scripts

Scripts for deploying NovaCron on NVIDIA Jetson Thor (Tegra) platform.

## Platform Requirements

- **Platform**: NVIDIA Jetson Thor (Tegra)
- **JetPack**: 7.4+
- **CUDA**: 13+
- **TensorRT**: Required for AI features

## Quick Start

```bash
# Full setup (first time)
./setup.sh all

# Start services
./start-services.sh

# Check health
./health-check.sh

# Stop services
./stop-services.sh
```

## Scripts

### setup.sh

Main setup orchestrator. Installs dependencies, configures Docker containers, builds backend/frontend.

```bash
# Run all steps
./setup.sh all

# Run individual steps
./setup.sh platform     # Check platform compatibility
./setup.sh deps         # Install dependencies
./setup.sh containers   # Setup Docker containers
./setup.sh env          # Create environment file
./setup.sh backend      # Build backend
./setup.sh frontend     # Build frontend
./setup.sh services     # Create systemd services
```

### start-services.sh

Starts NovaCron services in the correct order.

```bash
./start-services.sh all        # Start everything
./start-services.sh containers # Start Docker containers only
./start-services.sh backend    # Start backend only
./start-services.sh frontend   # Start frontend only
./start-services.sh status     # Show status
```

### stop-services.sh

Gracefully stops NovaCron services.

```bash
./stop-services.sh all        # Stop everything
./stop-services.sh frontend   # Stop frontend only
./stop-services.sh backend    # Stop backend only
./stop-services.sh containers # Stop Docker containers only
```

### health-check.sh

Comprehensive health verification.

```bash
./health-check.sh
```

Checks:
- GPU availability and memory
- Disk and memory usage
- Docker container status
- Database connectivity
- API and WebSocket endpoints
- Frontend availability

## Service Ports

| Service | Port | Notes |
|---------|------|-------|
| PostgreSQL | 15432 | Non-standard to avoid conflicts |
| Redis | 16379 | Non-standard to avoid conflicts |
| Qdrant | 16333 | Non-standard to avoid conflicts |
| API | 8090 | REST API |
| WebSocket | 8091 | Real-time events |
| Frontend | 8092 | Next.js web UI |

## Environment Variables

Override defaults via environment:

```bash
export POSTGRES_PORT=15432
export REDIS_PORT=16379
export QDRANT_PORT=16333
export API_PORT=8090
export WS_PORT=8091
export FRONTEND_PORT=8092
export NOVACRON_HOME=/home/kp/repos/novacron
```

## Systemd Services

After running `./setup.sh services`:

```bash
# Start on boot
sudo systemctl enable novacron-api novacron-frontend

# Manual control
sudo systemctl start novacron-api
sudo systemctl start novacron-frontend
sudo systemctl status novacron-api
sudo systemctl status novacron-frontend

# View logs
journalctl -u novacron-api -f
journalctl -u novacron-frontend -f
```

## Troubleshooting

### Port Conflicts

If ports are in use, set alternative ports:

```bash
export API_PORT=8190
export FRONTEND_PORT=8192
./start-services.sh
```

### Docker Container Issues

```bash
# View container logs
docker logs novacron-postgres
docker logs novacron-redis
docker logs novacron-qdrant

# Restart containers
docker restart novacron-postgres novacron-redis novacron-qdrant
```

### Build Failures

```bash
# Clean and rebuild backend
cd /home/kp/repos/novacron/backend
go clean -cache
go mod tidy
go build ./...

# Clean and rebuild frontend
cd /home/kp/repos/novacron/frontend
rm -rf node_modules .next
npm install
npm run build
```

## CUDA/TensorRT Verification

```bash
# Check CUDA
nvcc --version
nvidia-smi

# Check TensorRT
ldconfig -p | grep libnvinfer

# Test CUDA from Python
python3 -c "import torch; print(torch.cuda.is_available())"
```
