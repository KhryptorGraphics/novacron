#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "== YAML parse =="
node <<'JS'
const fs = require('fs');
const YAML = require('yaml');
const paths = [
  'deployment/profiles/docker-compose.canonical.yml',
  'deployment/docker/docker-compose.prod.yml',
  'deployment/production/docker-compose.production.yml',
  'deployment/kubernetes/configmap.yaml',
  'deployment/kubernetes/deployments.yaml',
  'deployment/kubernetes/services.yaml',
  'deployment/kubernetes/production-manifests.yaml',
];

for (const path of paths) {
  YAML.parseAllDocuments(fs.readFileSync(path, 'utf8')).forEach((document) => {
    if (document.errors.length > 0) {
      throw new Error(`${path}: ${document.errors.map((error) => error.message).join('; ')}`);
    }
  });
  console.log(`ok ${path}`);
}
JS

echo "== Docker compose config =="
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  docker compose --env-file deployment/profiles/local.env.example \
    -f deployment/profiles/docker-compose.canonical.yml --profile local config >/dev/null
  docker compose -f deployment/profiles/docker-compose.canonical.yml --profile single-node config >/dev/null
  docker compose -f deployment/profiles/docker-compose.canonical.yml --profile multi-node config >/dev/null
  echo "ok compose profiles"
else
  echo "skip compose profiles: docker compose unavailable"
fi

echo "== Canonical runtime build graph =="
(cd backend/core && go test ./cmd/novacron)

echo "== Frontend release gate =="
npm run typecheck:rc --prefix frontend

echo "phase5 validation ok"
