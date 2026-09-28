# NovaCron Comprehensive Deployment Guide

## Canonical entry points

- Bare metal / systemd: [../deploy/README.md](../deploy/README.md) (install-node.sh,
  the two systemd units, health checks, migration sequencing, backups).
- Docker Compose: `docker-compose.yml` / `docker-compose.production.yml` at the
  repository root, built from `docker/*.Dockerfile`.
- Kubernetes: the manifests in `deployment/kubernetes/` (the only supported
  k8s tree), walked through in
  [deployment/PRODUCTION_DEPLOYMENT_GUIDE.md](deployment/PRODUCTION_DEPLOYMENT_GUIDE.md).
  All runtime configuration is environment variables (see
  `docs/CANONICAL_CONTRACT_MATRIX.md`); there are no config YAML files to edit.

This file exists because several older references still point here. All
previous content referenced files that do not exist in the repository
(e.g. `k8s/novacron-deployment.secure.yaml`, `k8s/disaster-recovery/`,
`backend/database/migrations/001_performance_optimization.sql`,
`scripts/security/setup-vault.sh`, `infrastructure/terraform`) and has been
removed to avoid misleading operators.
