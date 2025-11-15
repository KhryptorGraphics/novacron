# Multi-stage Dockerfile for Onboarding System
# Optimized for production with security best practices

# Stage 1: Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    git \
    ca-certificates \
    tzdata \
    make \
    gcc \
    musl-dev

# Set working directory
WORKDIR /build

# Copy go mod files
COPY backend/go.mod backend/go.sum ./
RUN go mod download && go mod verify

# Copy source code
COPY backend/ ./

# Build arguments
ARG GO_VERSION
ARG BUILD_DATE
ARG VCS_REF

# Build the application with optimizations
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s -X main.Version=${VCS_REF} -X main.BuildDate=${BUILD_DATE}" \
    -a -installsuffix cgo \
    -o /build/bin/onboarding-server \
    ./cmd/onboarding/main.go

# Build migration tool
RUN go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest

# Stage 2: Runtime stage
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    curl \
    bash

# Create non-root user
RUN addgroup -g 1000 appuser && \
    adduser -D -u 1000 -G appuser appuser

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/bin/onboarding-server /app/onboarding-server
COPY --from=builder /go/bin/migrate /usr/local/bin/migrate

# Copy migrations
COPY backend/database/migrations /app/migrations

# Copy configuration files
COPY backend/config/onboarding.yaml /app/config/onboarding.yaml
COPY backend/config/beads-integration.yaml /app/config/beads-integration.yaml

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV APP_ENV=production \
    LOG_LEVEL=info \
    SERVER_PORT=8080 \
    METRICS_PORT=9090

# Labels
LABEL org.opencontainers.image.title="Novacron Onboarding System" \
      org.opencontainers.image.description="Advanced user onboarding with BEADS protocol integration" \
      org.opencontainers.image.version="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/novacron/onboarding" \
      org.opencontainers.image.vendor="Novacron"

# Entrypoint script
COPY deployments/docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/onboarding-server"]
