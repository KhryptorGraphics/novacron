# OpenCode Configuration

## Build/Test Commands
- **Build**: `make build` (Docker) or `docker-compose build`
- **Test All**: `make test` (Docker) or `make test-local` (local Go)
- **Test Single Package**: `go test -v ./backend/core/vm/...` or `go test -v ./backend/core/scheduler/policy/...`
- **Test Single File**: `go test -v ./backend/core/auth/auth_test.go`
- **Benchmarks**: `go test -bench=. ./backend/core/scheduler/policy/...`
- **Frontend Dev**: `cd frontend && npm run dev`
- **Frontend Lint**: `cd frontend && npm run lint`
- **Frontend Test**: `cd frontend && npm test`
- **Clean**: `make clean`

## Go Code Style
- Use `context.Context` for cancellation/timeouts, pass as first parameter
- Return errors explicitly, don't panic - use `if err != nil { return err }`
- Interface naming: add `-er` suffix (e.g., `VMManager`, `StorageProvider`)
- Package imports: stdlib first, then third-party, then local (separated by blank lines)
- Use structured logging with logrus: `log.WithFields(log.Fields{"vm_id": id}).Info("message")`
- Error wrapping: `fmt.Errorf("operation failed: %w", err)`
- Use `sync.Mutex` for thread safety, defer unlock immediately after lock

## TypeScript/React Style
- Use TypeScript interfaces for props and data structures
- Import UI components from `@/components/ui/` (shadcn/ui)
- Use `cn()` utility from `@/lib/utils` for conditional classes
- Prefer functional components with hooks over class components
- Use React Query (`@tanstack/react-query`) for data fetching
- File naming: kebab-case for components (`vm-list.tsx`)