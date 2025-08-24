# Build stage
FROM golang:1.23-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git gcc musl-dev

# Set working directory
WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy backend directory
COPY backend ./backend

# Build the application
RUN CGO_ENABLED=1 GOOS=linux go build -a -o novacron-hypervisor ./backend/core/cmd/novacron

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
