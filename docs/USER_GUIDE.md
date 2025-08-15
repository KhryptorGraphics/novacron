# NovaCron Quick‑Start Guide

## Prerequisites
- Docker & Docker‑Compose
- Go 1.22
- Node.js 20

## Development Setup
```bash
make test   # run all Go tests
cd frontend && npm ci && npm run dev
```

## Deployment
```bash
./scripts/deploy_production.sh   # systemd services, TLS, HA failover
```

## API Reference
See `config/novacron/api.yaml` for endpoint definitions.

## Monitoring
Dashboard available at `http://localhost:3000/dashboard`.

## Security
Run `go run ./security/hardening.go` to initialize secret management and audit logging.
