# NovaCron Orchestration Service Dockerfile
FROM golang:1.23-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata gcc musl-dev

# Set working directory
WORKDIR /app

# Copy go mod files
COPY backend/core/orchestration/go.mod backend/core/orchestration/go.sum ./
RUN go mod download

# Copy source code
COPY backend/core/orchestration/ ./

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o orchestration-service .

# Production image
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata curl bash

# Create non-root user
RUN adduser -D -s /bin/bash orchestrator

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/orchestration-service .

# Copy configuration files
COPY deployment/orchestration/config/ ./config/

# Create necessary directories
RUN mkdir -p /var/lib/novacron/orchestration /var/lib/novacron/ml-models /var/log/novacron \
    && chown -R orchestrator:orchestrator /var/lib/novacron /var/log/novacron /app

# Switch to non-root user
USER orchestrator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8443 9090

# Set environment variables
ENV GIN_MODE=release
ENV LOG_LEVEL=info
ENV CONFIG_PATH=/app/config/orchestration.yaml

# Start the service
CMD ["./orchestration-service"]