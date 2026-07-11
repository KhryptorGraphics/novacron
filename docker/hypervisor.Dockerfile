# Build stage
# NOTE: backend/core/go.mod requires go >= 1.24.0 (toolchain go1.24.6);
# golang:1.23-alpine cannot satisfy that with GOTOOLCHAIN=local (the default
# in a network-restricted build), so this must track backend/core's minimum.
FROM golang:1.25-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git gcc musl-dev

# Set working directory to the backend/core module root. backend/core has its
# own go.mod (module github.com/khryptorgraphics/novacron/backend/core) and
# is therefore NOT part of the root module's package graph — building
# "./backend/core/cmd/novacron" from /app (root module context) silently
# excludes it. Building from inside the module fixes that.
WORKDIR /app/backend/core

# Copy go mod and sum files for this module first (better layer caching)
COPY backend/core/go.mod backend/core/go.sum ./

# Download dependencies
RUN go mod download

# Copy the rest of the module
COPY backend/core ./

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -a -o /app/novacron-hypervisor ./cmd/novacron

# Final stage
FROM alpine:3.17

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libvirt \
    libvirt-client \
    qemu-system-x86_64 \
    qemu-img \
    dbus \
    polkit \
    openssh-client \
    iptables \
    iproute2 \
    procps \
    util-linux \
    virt-manager

# Create novacron user
RUN addgroup -S novacron && adduser -S novacron -G novacron

# Create necessary directories
RUN mkdir -p /var/lib/novacron/vms /etc/novacron /var/run/novacron
RUN chown -R novacron:novacron /var/lib/novacron /etc/novacron /var/run/novacron

# Copy binary from builder
COPY --from=builder /app/novacron-hypervisor /usr/local/bin/

# Add libvirt group to novacron user for KVM access
RUN adduser novacron libvirt

# Copy entrypoint script
COPY docker/hypervisor-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/hypervisor-entrypoint.sh

# Expose ports
EXPOSE 9000

# Set user
USER novacron

# Set environment variables
ENV NODE_ID=node1 \
    LOG_LEVEL=info \
    STORAGE_PATH=/var/lib/novacron/vms

# Set entrypoint
ENTRYPOINT ["hypervisor-entrypoint.sh"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:9000/health || exit 1

# Default command
CMD ["novacron-hypervisor", "--config", "/etc/novacron/config.yaml"]
