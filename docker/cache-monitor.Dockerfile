# Cache Monitor Dockerfile for NovaCron

FROM golang:1.23-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    git \
    ca-certificates \
    tzdata

# Set working directory
WORKDIR /app

# Copy go module files
COPY backend/core/go.mod backend/core/go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY backend/core/ ./

# Build the cache monitor binary
RUN CGO_ENABLED=0 GOOS=linux go build \
    -a -installsuffix cgo \
    -ldflags="-w -s" \
    -o cache-monitor \
    ./cmd/cache-monitor/

# Create final image
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    curl

# Create non-root user
RUN addgroup -g 1001 cache && \
    adduser -D -u 1001 -G cache cache

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/cache-monitor .

# Copy configuration files
COPY configs/ ./configs/

# Set ownership
RUN chown -R cache:cache /app

# Switch to non-root user
USER cache

# Expose ports
EXPOSE 9091 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9091/health || exit 1

# Default command
CMD ["./cache-monitor"]