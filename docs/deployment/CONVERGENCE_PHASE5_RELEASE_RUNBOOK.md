# Convergence Phase 5 Release Runbook

## Release Scope

Phase 5 ships the converged NovaCron runtime as a trusted deployment profile set. The canonical runtime is `backend/core/cmd/novacron`; duplicate legacy entrypoints stay compatibility-only until explicitly rewired.

## Supported Profiles

| Profile | Entry point | Purpose | Required backing services |
| --- | --- | --- | --- |
| `local` | `deployment/profiles/docker-compose.canonical.yml` | Developer/operator smoke deployment | bundled Postgres and Redis |
| `single-node` | `deployment/profiles/docker-compose.canonical.yml` | One runtime node with operator-managed dependencies | external Postgres/Redis optional |
| `multi-node` | `deployment/profiles/docker-compose.canonical.yml` | Trusted two-node seeded-discovery fabric | shared Postgres/Redis recommended |
| `kubernetes` | `deployment/kubernetes/*.yaml` | Production cluster profile | Kubernetes secrets/configmaps, Postgres, Redis |

## Preflight

```bash
npm run deploy:validate
npm run ci:canonical --prefix frontend
npm run test:e2e:canonical -- --reporter=list
```

Required pass criteria:

- Docker Compose profiles render with `docker compose config`.
- Kubernetes manifests parse as YAML.
- `backend/core/cmd/novacron` package tests pass.
- Frontend RC typecheck passes.
- Canonical E2E passes before release-candidate tag.

## Local Smoke

```bash
docker compose --env-file deployment/profiles/local.env.example \
  -f deployment/profiles/docker-compose.canonical.yml --profile local up --build
```

Check:

- API health: `curl http://localhost:8091/health`
- Frontend: `http://localhost:8080`
- Metrics bound to loopback: `http://localhost:9090/metrics`

## Single Node

Set `NOVACRON_DATABASE_URL` and `NOVACRON_REDIS_URL` when dependencies are host-managed.

```bash
NOVACRON_DATABASE_URL=postgres://user:pass@db:5432/novacron?sslmode=require \
NOVACRON_REDIS_URL=redis://:pass@redis:6379/0 \
npm run deploy:single-node
```

## Multi Node

Use seeded discovery only inside a trusted network.

```bash
NOVACRON_DISCOVERY_SEEDS=http://novacron-node-a:8091/api/v1/federation/inventory \
npm run deploy:multi-node
```

Check:

- Both nodes expose `/health`.
- Node inventory signatures verify.
- Placement and migration remain on cold mode unless latency gates are explicitly enabled.

## Kubernetes

Apply order:

```bash
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/rbac.yaml
kubectl apply -f deployment/kubernetes/secrets.yaml
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/services.yaml
kubectl apply -f deployment/kubernetes/deployments.yaml
```

Verify:

```bash
kubectl -n novacron rollout status deploy/novacron-api
kubectl -n novacron rollout status deploy/novacron-frontend
kubectl -n novacron get pods,svc
```

## Rollback

- Compose: `docker compose -f deployment/profiles/docker-compose.canonical.yml down`, then restart previous image tag with `VERSION=<previous>`.
- Kubernetes: `kubectl -n novacron rollout undo deploy/novacron-api` and `kubectl -n novacron rollout undo deploy/novacron-frontend`.
- Data: restore Postgres/volume backups before re-enabling migration or backup workers if schema/data corruption suspected.

## Release Gate

Do not tag release candidate until all are true:

- `npm run deploy:validate` passes.
- Canonical frontend CI and E2E pass.
- GitNexus `detect_changes` reviewed.
- No untracked generated deployment artifacts.
- Beads for Phase 5 are closed or linked as release-blocking follow-up.
