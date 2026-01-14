CRUSH guidelines for agentic contributors

Build & test commands
- Backend (Go):
  - Run all tests: cd backend/core && go test ./...
  - Run a single package: cd backend/core/vm && go test -v ./...
  - Run a single test function in package: cd backend/core/vm && go test -run TestName -v
  - Makefile helper: make test (runs recommended test suite)
- Frontend (Next.js):
  - Install: cd frontend && npm install
  - Dev server: cd frontend && npm run dev
  - Build: cd frontend && npm run build
  - Lint: cd frontend && npm run lint
  - Run a single Jest test file: cd frontend && npm test -- <path-to-test-file>
- Docker: docker-compose -f docker-compose.dev.yml up -d (dev stack)

Code style & conventions
- Formatting: Use gofmt / goimports for Go. Run gofmt -w and goimports -w before commits.
- Imports: Group stdlib, third-party, internal; use goimports to auto-fix ordering.
- Errors: Return errors (no panic). Wrap with fmt.Errorf("...: %w", err) when adding context. Prefer sentinel errors when appropriate.
- Context: Use context.Context as first arg for cancellable ops and pass down contexts.
- Naming: camelCase for unexported, PascalCase for exported types/functions. Test names start with TestXxx.
- Types: Prefer small concrete structs; use interfaces for behavior contracts and to aid testing.
- Logging: Use structured logs with levels; include IDs/correlation, avoid logging secrets.
- Tests: Keep unit tests fast and deterministic; use mocks and testdata/ for external dependencies.
- Concurrency: Prefer channels and contexts; protect shared state with mutexes and keep goroutines cancellable.
- Security: Do not commit secrets. Use .env and .env.template for local config.

Repository agent rules
- Use .crush/ for ephemeral agent state (ignored by git).
- Cursor/Copilot rules: no .cursor or .cursorrules or .github/copilot-instructions.md found; if added, summarize rules here.

Developer notes
- Run linters and type checks before PRs (Go: golangci-lint or go vet; Frontend: npm run lint).
- If adding env variables, update .env.template.
- Keep commits focused and include test plan for behavior changes.

💘 Generated with Crush
