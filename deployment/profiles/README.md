# NovaCron Canonical Deployment Profiles

This directory is the release-candidate deployment entrypoint for the converged runtime.

## Profiles

- `local`: frontend, canonical runtime, Postgres, and Redis on one Docker host.
- `single-node`: canonical runtime only, with database/cache supplied by the operator.
- `multi-node`: two canonical runtime nodes using seeded discovery.

## Commands

```bash
npm run deploy:validate
docker compose --env-file deployment/profiles/local.env.example -f deployment/profiles/docker-compose.canonical.yml --profile local config
docker compose --env-file deployment/profiles/local.env.example -f deployment/profiles/docker-compose.canonical.yml --profile local up --build
docker compose -f deployment/profiles/docker-compose.canonical.yml --profile single-node up --build
docker compose -f deployment/profiles/docker-compose.canonical.yml --profile multi-node up --build
```

## Runtime Contract

All profiles pass the same runtime-manifest environment contract consumed by `backend/core/cmd/novacron`:

- `NOVACRON_RUNTIME_MANIFEST_VERSION`
- `NOVACRON_DEPLOYMENT_PROFILE`
- `NOVACRON_DISCOVERY_MODE`
- `NOVACRON_FEDERATION_MODE`
- `NOVACRON_MIGRATION_MODE`
- `NOVACRON_AUTH_MODE`
- `NOVACRON_STORAGE_CLASSES`
- `NOVACRON_ENABLED_SERVICES`
- `NOVACRON_DISCOVERY_SEEDS`

Legacy deployment assets remain for historical compatibility, but release work should use these profiles first.
