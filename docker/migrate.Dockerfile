# NovaCron DB Migration Tool
#
# Builds database/migrate.go, which is its OWN Go module (database/go.mod) —
# nothing in the stack invoked it before this. This image is consumed as:
#   - a docker-compose "migrate" init service (see docker-compose.yml), which
#     runs before the api service starts and also publishes the built binary
#     onto a shared volume so docker/api-entrypoint.sh can defensively re-run
#     it (idempotent: golang-migrate's Up() is a no-op when already current).
#   - a Kubernetes Job (see k8s/migrate-job.yaml) run before/alongside the
#     novacron-api Deployment.

# Build stage
FROM golang:1.23-alpine AS builder

RUN apk add --no-cache git

WORKDIR /app/database

# Copy module files first for better layer caching
COPY database/go.mod database/go.sum ./
RUN go mod download

# Copy the rest of the migration tool (migrate.go + embedded migrations/*.sql)
COPY database ./

RUN CGO_ENABLED=0 GOOS=linux go build -a -o /app/bin/migrate-db .

# Final stage
FROM alpine:3.19

RUN apk add --no-cache ca-certificates && \
    addgroup -g 1000 -S novacron && adduser -u 1000 -S novacron -G novacron && \
    mkdir -p /shared/bin && chown -R novacron:novacron /shared/bin

COPY --from=builder /app/bin/migrate-db /usr/local/bin/migrate-db
COPY docker/migrate-entrypoint.sh /usr/local/bin/migrate-entrypoint.sh
RUN chmod +x /usr/local/bin/migrate-entrypoint.sh /usr/local/bin/migrate-db

USER novacron

ENTRYPOINT ["/usr/local/bin/migrate-entrypoint.sh"]

# Default direction; override with e.g. `-direction=down` for k8s Job args.
CMD ["-direction=up"]
