# NovaCron Quick‑Start Guide

## Prerequisites
- Docker & Docker‑Compose
- Go 1.22
- Node.js 20

## Development Setup
```bash
make test   # run all Go tests
go mod tidy   # one-time module sync for the Go backend
npm install
cd frontend && npm ci
npm run dev   # repo root: standalone Go API (main_working.go) on :8090 + frontend on :8092
```

If you only need the frontend:
```bash
cd frontend && npm ci && npm run dev
```

## Deployment
```bash
./scripts/deploy_production.sh   # systemd services, TLS, HA failover
```

## API Reference
See `config/novacron/api.yaml` for endpoint definitions.

## Monitoring
Dashboard available at `http://localhost:8092/dashboard`.

## Security
Run `go run ./security/hardening.go` to initialize secret management and audit logging.
