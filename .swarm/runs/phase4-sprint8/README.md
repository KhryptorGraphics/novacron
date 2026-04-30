# Phase 4 Sprint 8: Frontend API Client Consolidation

## Status

Complete.

## Implemented Slice

- Created `novacron-3wz` from Phase 4 epic `NC-p4a`.
- Added a typed automation API module for jobs and workflows under `frontend/src/lib/api/automation.ts`.
- Added React Query-backed automation hooks under `frontend/src/lib/api/hooks/useAutomation.ts`.
- Repointed dashboard job and workflow components away from legacy `@/hooks/useAPI`.
- Repointed the scheduling dashboard websocket status to the existing canonical websocket hook.
- Added API and hook barrel exports for the automation client surface.
- Updated dashboard component tests to mock the new typed automation hook path.

## Current Boundary

- This slice is intentionally limited to the dashboard scheduling/job/workflow surface.
- Remaining Phase 4 work still includes admin VM mock data, user/database mock data, flow demo data, network configuration demo data, orchestration direct fetches, and broader E2E parity.
- Follow-up beads filed: `novacron-hig` for remaining mock/demo surfaces and `novacron-ab5` for orchestration direct fetch consolidation.

## Validation

- `git diff --check`: pass.
- `npm test -- --runTestsByPath tests/components/dashboard.test.tsx --runInBand --coverage=false`: pass; existing Radix/React `act(...)` warnings remain.
- `npm run test:smoke -- --runInBand`: pass, 13 suites / 24 tests.
- `npx tsc --noEmit`: fail on repo-wide pre-existing frontend type debt; no diagnostics remained for this slice's changed files after narrowing the new API barrel.
