# Sprint 1 Toolchain And CI Baseline

## Supported Defaults

| Surface | Canonical source | Selected default |
| --- | --- | --- |
| Go core platform lanes | `go.mod` | `go 1.24.0` with `toolchain go1.24.6` |
| Node frontend and Playwright lanes | `.nvmrc` | `20` |
| JavaScript package manager | `package-lock.json`, `frontend/package-lock.json` | `npm` |

## Canonical Automatic Workflows

- `.github/workflows/ci.yml`
- `.github/workflows/integration-tests.yml`
- `.github/workflows/e2e-tests.yml`

## Manual-Only Specialized Workflows

- `.github/workflows/ci-cd.yml`
- `.github/workflows/ci-cd-production.yml`
- `.github/workflows/dwcp-v3-ci.yml`
- `.github/workflows/dwcp-v3-cd.yml`
- `.github/workflows/e2e-visual-regression.yml`

## Canonical Commands

### Backend

```bash
go test ./backend/cmd/api-server ./backend/api/graphql ./backend/api/security ./backend/api/websocket
go build ./backend/cmd/api-server
```

### Frontend

```bash
cd frontend
npm ci
npm run ci:canonical
```

The canonical frontend lane intentionally covers smoke tests plus production build. Repo-wide lint remains outside the Sprint 1 gate because the current workspace has unresolved legacy lint debt and Next 16 no longer supports the previous `next lint` invocation.

### Canonical E2E Smoke

```bash
npm ci
cd frontend && npm ci
PLAYWRIGHT_BASE_URL=http://127.0.0.1:3000 PLAYWRIGHT_SKIP_AUTH=true npm run test:e2e:canonical
```

The retained smoke fixture now seeds the same auth session shape the frontend bootstrap expects by mocking `/api/auth/me` and preloading the corresponding local-storage keys.

## Deferred Drift

- Specialized onboarding, DWCP phase deployment, and scheduled comprehensive workflows still carry independent toolchain pins and should be normalized only when they are promoted back into the shipping path.
- `tools/indexer/` retains its own legacy Yarn lockfile and pnpm metadata; it is explicitly outside the Sprint 1 default package-manager baseline.
