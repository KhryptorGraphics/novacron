# Go test runner for backend/core (docker-compose.test.yml, profile `testing`).
# glibc (bookworm), NOT alpine: `-race` in GO_TEST_FLAGS does not work on musl.
# The compose file bind-mounts ./backend to /app, so /app/core is the
# backend/core module; its deps (e.g. backend/pkg/logger pseudo-versions)
# resolve from the module proxy and are cached in the go_mod_cache volume.
FROM golang:1.25-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/core

ENV CGO_ENABLED=1 \
    GOFLAGS=-mod=mod

# Run the short test census; the service depends_on api-server/postgres/
# redis-master health, so integration-style tests can reach them.
CMD ["sh", "-c", "mkdir -p /test-results && go test -short ${GO_TEST_FLAGS:-} ./... 2>&1 | tee /test-results/go-test.log"]