# NovaCron User Guide

## Prerequisites

- Go **1.25.0** or newer (see `go.mod`)
- PostgreSQL 15+ reachable through a `DB_URL` connection string
- A strong `AUTH_SECRET` (>= 32 random hex chars)

## Quick start

```bash
# Database
make db-migrate            # applies backend/database/migrations/*.up.sql

# Canonical API server
make serve                  # go run ./backend/cmd/api-server (port 8090)

# Frontend
npm run start:frontend     # Next.js on port 8092 (same-origin by default)
```

## Features

- VM CRUD, migration (sync/async), fabric compute jobs, drain, transfers
- Org-scoped auth (admin/super-admin bypass) on /api/* and /api/v1/*
- Security operations: scan, compliance, RBAC, 2FA, audit export
- GraphQL at /graphql for volume-aware queries
- WebSockets: metrics, alerts, logs, console, security events

Full endpoint inventory is in [CANONICAL_CONTRACT_MATRIX.md](CANONICAL_CONTRACT_MATRIX.md).
Deployment is described in [../deploy/README.md](../deploy/README.md).
