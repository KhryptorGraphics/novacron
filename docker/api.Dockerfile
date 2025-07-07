# Build stage for Go API
FROM golang:1.19-alpine AS go-builder

# Install build dependencies
RUN apk add --no-cache git gcc musl-dev

# Set working directory
WORKDIR /app

# Copy go mod and sum files
COPY backend/core/go.mod backend/core/go.sum ./

# Download dependencies
RUN go mod download

# Copy Go source code
COPY backend/core ./

# Build the API service
RUN CGO_ENABLED=0 GOOS=linux go build -a -o novacron-api ./cmd/api

# Python build stage
FROM python:3.9-slim AS py-builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY backend/services/requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create novacron user
RUN groupadd -r novacron && useradd -r -g novacron novacron

# Create necessary directories
RUN mkdir -p /etc/novacron /var/run/novacron /var/log/novacron
RUN chown -R novacron:novacron /etc/novacron /var/run/novacron /var/log/novacron

# Copy Go binary from builder
COPY --from=go-builder /app/novacron-api /usr/local/bin/

# Copy Python virtual environment
COPY --from=py-builder /venv /venv

# Copy Python API service
COPY backend/services/api /app/api
COPY backend/services/common /app/common

# Expose ports
EXPOSE 8090

# Set user
USER novacron

# Set environment variables
ENV PATH="/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    LOG_LEVEL=info \
    API_PORT=8090

# Add entrypoint script
COPY docker/api-entrypoint.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/api-entrypoint.sh
USER novacron

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:8090/api/health || exit 1

# Set entrypoint
ENTRYPOINT ["api-entrypoint.sh"]

# Default command
CMD ["novacron-api", "--config", "/etc/novacron/config.yaml"]
