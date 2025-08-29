# Multi-stage build for NovaCron Edge Agent
# Optimized for resource-constrained edge devices

# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the edge agent binary
# Use build flags to minimize binary size
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-w -s -X main.version=$(git describe --tags --always)" \
    -a -installsuffix cgo \
    -o edge-agent \
    ./cmd/edge-agent

# Multi-architecture support
FROM golang:1.21-alpine AS builder-arm64
RUN apk add --no-cache git ca-certificates tzdata
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build \
    -ldflags="-w -s -X main.version=$(git describe --tags --always)" \
    -a -installsuffix cgo \
    -o edge-agent-arm64 \
    ./cmd/edge-agent

FROM golang:1.21-alpine AS builder-arm
RUN apk add --no-cache git ca-certificates tzdata
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=arm GOARM=7 go build \
    -ldflags="-w -s -X main.version=$(git describe --tags --always)" \
    -a -installsuffix cgo \
    -o edge-agent-arm \
    ./cmd/edge-agent

# Runtime stage - minimal image for edge deployment
FROM scratch AS runtime

# Import timezone data and certificates
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Create non-root user
COPY --from=builder /etc/passwd /etc/passwd

# Copy the binary
COPY --from=builder /app/edge-agent /edge-agent

# Default configuration
COPY edge-agent.yaml /etc/novacron/edge-agent.yaml

# Create necessary directories
COPY --from=builder --chown=nobody /tmp /tmp
USER nobody

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["/edge-agent", "health-check"]

# Set entrypoint
ENTRYPOINT ["/edge-agent"]
CMD ["--config", "/etc/novacron/edge-agent.yaml"]

# Multi-architecture runtime images
FROM runtime AS runtime-amd64
COPY --from=builder /app/edge-agent /edge-agent

FROM runtime AS runtime-arm64
COPY --from=builder-arm64 /app/edge-agent-arm64 /edge-agent

FROM runtime AS runtime-arm
COPY --from=builder-arm /app/edge-agent-arm /edge-agent

# Development image with shell and debugging tools
FROM alpine:3.18 AS development

# Install debugging and development tools
RUN apk add --no-cache \
    bash \
    curl \
    htop \
    strace \
    tcpdump \
    net-tools \
    redis \
    ca-certificates

# Copy binary
COPY --from=builder /app/edge-agent /usr/local/bin/edge-agent

# Copy configuration
COPY edge-agent.yaml /etc/novacron/edge-agent.yaml

# Create directories
RUN mkdir -p /var/log/novacron /var/lib/novacron /etc/novacron/certs

# Create novacron user
RUN addgroup -g 1000 novacron && \
    adduser -D -s /bin/bash -u 1000 -G novacron novacron

# Set ownership
RUN chown -R novacron:novacron /var/log/novacron /var/lib/novacron

# Switch to novacron user
USER novacron

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Development entrypoint
ENTRYPOINT ["/usr/local/bin/edge-agent"]
CMD ["--config", "/etc/novacron/edge-agent.yaml", "--log-level", "debug"]

# Default runtime image (minimal production)
FROM runtime-amd64 AS default