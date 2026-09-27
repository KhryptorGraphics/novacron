#!/bin/sh
# NovaCron Hypervisor Entrypoint Script
#
# backend/core/cmd/novacron is flag-driven (-config, -node-id, -data-dir,
# -listen). Its optional -config YAML uses the `storage:`/`hypervisor:`/
# `vm_manager:`/`scheduler:`/`auth:` schema of runtimeConfigFile (main.go);
# when the file is absent the built-in defaults apply, so this entrypoint no
# longer fabricates one. Environment (all optional):
#   NODE_ID          node id (default: container hostname)
#   STORAGE_PATH     data dir (default /var/lib/novacron/vms)
#   HYPERVISOR_LISTEN listen address (default 0.0.0.0:9000 — the port the
#                    api-server dials via HYPERVISOR_ADDRS, and the port the
#                    image's HEALTHCHECK probes at /healthz)
#   NOVACRON_AUTH_POSTGRES_URL / NOVACRON_AUTH_REDIS_URL / NOVACRON_TRUSTED_PROXIES
#                    read directly by the binary's runtime-auth stack
set -e

STORAGE_PATH="${STORAGE_PATH:-/var/lib/novacron/vms}"
HYPERVISOR_LISTEN="${HYPERVISOR_LISTEN:-0.0.0.0:9000}"

if [ ! -e /dev/kvm ]; then
    echo "WARNING: /dev/kvm not found. Hardware virtualization is unavailable;"
    echo "         KVM guests will not start (pass --device /dev/kvm)."
fi

if [ ! -d "$STORAGE_PATH" ]; then
    echo "Creating VM storage directory $STORAGE_PATH..."
    mkdir -p "$STORAGE_PATH"
fi

echo "Starting NovaCron Hypervisor..."
echo "Node ID: ${NODE_ID:-$(hostname)}"
echo "Storage Path: $STORAGE_PATH"
echo "Listen: $HYPERVISOR_LISTEN"
echo "Cluster Address: ${CLUSTER_ADDR:-novacron-api:8090}"

# Default command: run the binary with the flags derived above. Any explicit
# CMD/args override this entirely.
if [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ "$1" = "novacron-hypervisor" ]; }; then
    set -- novacron-hypervisor \
        -config /etc/novacron/config.yaml \
        -data-dir "$STORAGE_PATH" \
        -listen "$HYPERVISOR_LISTEN"
    if [ -n "${NODE_ID:-}" ]; then
        set -- "$@" -node-id "$NODE_ID"
    fi
fi

exec "$@"
