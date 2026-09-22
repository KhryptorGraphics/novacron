#!/usr/bin/env bash
# NovaCron node install: binaries, secrets, database, systemd units, health.
#
# Installs one NovaCron *role* on this host. Both roles run the same binary
# (backend/cmd/api-server); they differ in the port they serve and in what the
# operator expects of the node:
#
#   control  the API/fabric control plane on 8090 (accepts joins, serves users)
#   compute  a fabric compute node on 9000 (runs jobs and VMs, joins a seed)
#
# Everything the service reads at runtime lives in /etc/novacron/novacron.env
# (mode 0600, root-owned); the fabric shared secret is also kept as its own
# file, /etc/novacron/migration.secret, so it can be copied to the next node
# without handing over the rest of the environment.
#
# Usage:  sudo deploy/scripts/install-node.sh --db-url URL [options]
# Exit:   0 = installed (and started, unless --no-start), 1 = failed,
#         2 = usage error.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_DIR="/etc/systemd/system"
OPT_DIR="/opt/novacron"
BIN_DIR="$OPT_DIR/bin"
CONF_DIR="/etc/novacron"
ENV_FILE="$CONF_DIR/novacron.env"
SECRET_FILE="$CONF_DIR/migration.secret"
VAR_DIR="/var/lib/novacron"
LOG_DIR="/var/log/novacron"
SERVICE_USER="novacron"

ROLE="control"
DB_URL_ARG=""
NODE_ID=""
JOIN_ADDR=""
JOIN_PEERS=""
API_PORT=""
STORAGE_PATH="$VAR_DIR/vms"
PREBUILT=""
SKIP_BUILD=0
NO_START=0
CREATE_DB=0
FORCE_ENV=0

log()  { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
warn() { printf '%s WARNING: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
die()  { printf '%s ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 1; }

usage() {
  sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --db-url URL        postgres URL for the control-plane database (required on
                      first install; ignored once the env file exists)
  --role control|compute
                      which unit this host runs (default control). compute
                      writes NOVACRON_JOIN_ADDR for port 9000; control for 8090.
  --node-id ID        cluster node id (default: short hostname). Must be unique
                      across the fabric and must not be "local".
  --join-addr HOST:PORT
                      addr this node advertises to peers (default: first
                      non-loopback IPv4 + the role port)
  --join-peers LIST   comma-separated seed addrs to join at boot, e.g.
                      10.0.0.1:8090 (also settable later by join-cluster.sh)
  --api-port PORT     write API_PORT into the env file, overriding the unit's
                      role default for every NovaCron unit on this host
  --storage-path DIR  VM storage root (default /var/lib/novacron/vms)
  --binary PATH       install this prebuilt api-server instead of building from
                      source (the migrator is still built from ./database)
  --skip-build        reuse the binaries already in /opt/novacron/bin
  --create-db         create the database when it does not exist yet
  --force-env         rewrite the env file (rotates secrets; a running fabric
                      will reject joins until the secret is re-distributed)
  --no-start          install everything but do not enable/start the unit
  -h, --help          this text
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --db-url)       DB_URL_ARG="${2:?--db-url needs a value}"; shift 2 ;;
    --role)         ROLE="${2:?--role needs a value}"; shift 2 ;;
    --node-id)      NODE_ID="${2:?--node-id needs a value}"; shift 2 ;;
    --join-addr)    JOIN_ADDR="${2:?--join-addr needs a value}"; shift 2 ;;
    --join-peers)   JOIN_PEERS="${2:?--join-peers needs a value}"; shift 2 ;;
    --api-port)     API_PORT="${2:?--api-port needs a value}"; shift 2 ;;
    --storage-path) STORAGE_PATH="${2:?--storage-path needs a value}"; shift 2 ;;
    --binary)       PREBUILT="${2:?--binary needs a value}"; shift 2 ;;
    --skip-build)   SKIP_BUILD=1; shift ;;
    --create-db)    CREATE_DB=1; shift ;;
    --force-env)    FORCE_ENV=1; shift ;;
    --no-start)     NO_START=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *)              usage >&2; die "unknown argument: $1" ;;
  esac
done

case "$ROLE" in
  control) ROLE_PORT=8090; ROLE_UNIT="novacron-api-server.service"; OTHER_UNIT="novacron-fabric-peer.service" ;;
  compute) ROLE_PORT=9000; ROLE_UNIT="novacron-fabric-peer.service"; OTHER_UNIT="novacron-api-server.service" ;;
  *)       die "--role must be control or compute, got '$ROLE'" ;;
esac
[ -z "$API_PORT" ] && API_PORT="$ROLE_PORT"
case "$API_PORT" in ''|*[!0-9]*) die "--api-port must be a number, got '$API_PORT'";; esac

# --- root and prerequisites -------------------------------------------------

[ "$(id -u)" = "0" ] || die "must run as root: sudo $0 $*"

req() { # req <binary> <what needs it and how to install it>
  command -v "$1" >/dev/null 2>&1 || MISSING+=("$1: $2")
}

ARCH_QEMU="qemu-system-$(uname -m | sed 's/x86_64/x86_64/; s/aarch64/aarch64/')"
MISSING=()
req psql         "postgres client — the migrations run through it: apt-get install -y postgresql-client"
req "$ARCH_QEMU" "the hypervisor binary — guests cannot boot without it: apt-get install -y qemu-system-$(uname -m)"
req qemu-img     "disk image handling for VM create/convert/migrate: apt-get install -y qemu-utils"
req socat        "peer/postgres plumbing in tests and cross-node debugging: apt-get install -y socat"
req tc           "link shaping (the fabric's measured RTT/throughput work is meaningless unshaped): apt-get install -y iproute2"
req openssl      "HMAC signing for join-cluster.sh and secret generation: apt-get install -y openssl"
req curl         "post-install /health check and join verification: apt-get install -y curl"
if [ "$SKIP_BUILD" = "0" ] && [ -z "$PREBUILT" ]; then
  req go         "builds the api-server and migrator from source (or pass --binary/--skip-build): apt-get install -y golang-go"
fi
if [ "${#MISSING[@]}" -gt 0 ]; then
  printf '%s ERROR: missing required prerequisites:\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  printf '  - %s\n' "${MISSING[@]}" >&2
  exit 1
fi

if [ ! -e /dev/kvm ]; then
  warn "/dev/kvm is missing: guests will run under qemu TCG (an order of magnitude slower). Enable virtualization in firmware, 'modprobe kvm', and confirm the CPU exposes it."
fi

# --- service user and directories ------------------------------------------

if ! getent passwd "$SERVICE_USER" >/dev/null; then
  log "creating system user $SERVICE_USER"
  useradd --system --home-dir "$OPT_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi
# The units grant this group via SupplementaryGroups=; adding it to the account
# too keeps a manual `sudo -u novacron qemu-system-...` working.
if getent group kvm >/dev/null; then
  usermod -aG kvm "$SERVICE_USER"
else
  warn "no 'kvm' group on this host — /dev/kvm ownership may need a drop-in SupplementaryGroups= entry"
fi

install -d -o root       -g root       -m 0755 "$OPT_DIR" "$BIN_DIR"
install -d -o root       -g root       -m 0755 "$OPT_DIR/deploy"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$VAR_DIR" "$VAR_DIR/vms" "$VAR_DIR/volumes"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0750 "$LOG_DIR"
install -d -o root       -g root       -m 0750 "$CONF_DIR"
cp -a "$REPO_ROOT/deploy/README.md" "$REPO_ROOT/deploy/scripts" "$REPO_ROOT/deploy/systemd" "$OPT_DIR/deploy/"

# --- secrets and environment ------------------------------------------------

machine_ip() {
  ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -n 1
}

env_get() { sed -n "s/^$1=//p" "$ENV_FILE" | tail -n 1 | sed 's/^"\(.*\)"$/\1/'; }

if [ -e "$ENV_FILE" ] && [ "$FORCE_ENV" = "0" ]; then
  log "keeping existing $ENV_FILE (use --force-env to rotate secrets and rewrite it)"
  # Read back what the running node actually uses, so the checks below judge the
  # real configuration rather than this invocation's defaults.
  ENV_DB_URL="$(env_get DB_URL)"
  ENV_STORAGE="$(env_get STORAGE_PATH)"
  ENV_NODE_ID="$(env_get NOVACRON_NODE_ID)"
  ENV_JOIN_ADDR="$(env_get NOVACRON_JOIN_ADDR)"
  ENV_JOIN_PEERS="$(env_get NOVACRON_JOIN_PEERS)"
  ENV_API_PORT="$(env_get API_PORT)"
  if [ -n "$ENV_DB_URL" ]; then DB_URL_ARG="$ENV_DB_URL"; fi
  if [ -n "$ENV_STORAGE" ]; then STORAGE_PATH="$ENV_STORAGE"; fi
  if [ -z "$NODE_ID" ] && [ -n "$ENV_NODE_ID" ]; then NODE_ID="$ENV_NODE_ID"; fi
  if [ -z "$JOIN_ADDR" ] && [ -n "$ENV_JOIN_ADDR" ]; then JOIN_ADDR="$ENV_JOIN_ADDR"; fi
  if [ -z "$JOIN_PEERS" ] && [ -n "$ENV_JOIN_PEERS" ]; then JOIN_PEERS="$ENV_JOIN_PEERS"; fi
  if [ -n "$ENV_API_PORT" ]; then API_PORT="$ENV_API_PORT"; fi
fi

[ -n "$DB_URL_ARG" ] || die "--db-url is required on first install, e.g. --db-url 'postgresql://novacron:secret@127.0.0.1:5432/novacron?sslmode=disable'"

if [ -z "$NODE_ID" ]; then
  NODE_ID="$(hostname -s)"
fi
case "$NODE_ID" in
  local) die "node id 'local' is the api-server's fallback for an unconfigured node and collides with every other unconfigured node — pass --node-id" ;;
  ""|*)   : ;;
esac

HOST_IP="$(machine_ip)"
[ -n "$HOST_IP" ] || HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ -n "$HOST_IP" ] || HOST_IP="127.0.0.1"
if [ -z "$JOIN_ADDR" ]; then
  JOIN_ADDR="$HOST_IP:$API_PORT"
fi

if [ ! -e "$ENV_FILE" ] || [ "$FORCE_ENV" = "1" ]; then
  AUTH_SECRET="$(openssl rand -hex 32)"
  MIGRATION_SECRET="$(openssl rand -hex 32)"
  install -m 0600 -o root -g root /dev/null "$SECRET_FILE"
  printf '%s\n' "$MIGRATION_SECRET" > "$SECRET_FILE"
  install -m 0600 -o root -g root /dev/null "$ENV_FILE"
  {
    printf '# NovaCron node environment — read by %s and %s.\n' \
      "novacron-api-server.service" "novacron-fabric-peer.service"
    printf '# Secrets live here: keep it root-only (0600). Written by deploy/scripts/install-node.sh;\n'
    printf '# edit by hand for anything the installer does not cover, then systemctl restart the unit.\n\n'
    printf 'DB_URL=%s\n' "$DB_URL_ARG"
    printf 'AUTH_SECRET=%s\n' "$AUTH_SECRET"
    # Fail-closed fabric auth: an empty secret makes this node refuse every
    # peer RPC (migrations, jobs, capacity probes) instead of accepting any.
    printf 'NOVACRON_MIGRATION_SECRET=%s\n' "$MIGRATION_SECRET"
    printf 'NOVACRON_NODE_ID=%s\n' "$NODE_ID"
    printf 'NOVACRON_JOIN_ADDR=%s\n' "$JOIN_ADDR"
    if [ -n "$JOIN_PEERS" ]; then
      printf 'NOVACRON_JOIN_PEERS=%s\n' "$JOIN_PEERS"
    fi
    if [ -n "$API_PORT" ]; then
      printf '\n# Overrides the per-role port default (control 8090 / compute 9000) for\n'
      printf '# every NovaCron unit on this host. NOVACRON_JOIN_ADDR must match it.\n'
      printf 'API_PORT=%s\n' "$API_PORT"
    fi
    printf 'STORAGE_PATH=%s\n' "$STORAGE_PATH"
    printf 'LOG_LEVEL=info\n'
    printf 'LOG_FORMAT=json\n'
    printf '\n# Optional usage-metering rate card (defaults are 0 = recorded, unpriced).\n'
    printf '# NOVACRON_RATE_PER_VCpu_HOUR=0\n'
    printf '# NOVACRON_RATE_PER_GB_EGRESS=0\n'
    printf '# NOVACRON_RATE_PER_JOB_SECOND=0\n'
    printf '# NOVACRON_RATE_PER_MIGRATION=0\n'
  } > "$ENV_FILE"
  log "wrote $ENV_FILE and $SECRET_FILE (0600 root:root)"
else
  log "reusing $SECRET_FILE"
fi
chmod 0600 "$ENV_FILE" "$SECRET_FILE"

# --- binaries ---------------------------------------------------------------

if [ "$SKIP_BUILD" = "0" ]; then
  if [ -n "$PREBUILT" ]; then
    [ -x "$PREBUILT" ] || die "--binary $PREBUILT is not executable"
    install -m 0755 -o root -g root "$PREBUILT" "$BIN_DIR/api-server"
  else
    log "building api-server (CGO_ENABLED=1 go build ./backend/cmd/api-server)"
    ( cd "$REPO_ROOT" && CGO_ENABLED=1 go build -o "$BIN_DIR/api-server" ./backend/cmd/api-server )
  fi
  # ./database is its own module with the migrations embedded, so this binary
  # needs no network and no golang-migrate CLI at install time.
  log "building the migrator (cd database && go build .)"
  ( cd "$REPO_ROOT/database" && CGO_ENABLED=1 go build -o "$BIN_DIR/novacron-migrate" . )
else
  [ -x "$BIN_DIR/api-server" ] || die "--skip-build requested but $BIN_DIR/api-server does not exist"
  [ -x "$BIN_DIR/novacron-migrate" ] || die "--skip-build requested but $BIN_DIR/novacron-migrate does not exist"
fi
chmod 0755 "$BIN_DIR/api-server" "$BIN_DIR/novacron-migrate"

# --- database ---------------------------------------------------------------

maintenance_url() { # postgres URL of the same cluster's "postgres" database
  local url="$1" query="" base dbname
  case "$url" in *"://"*) : ;; *) die "DB_URL must look like postgresql://user:pass@host:port/dbname, got '$url'" ;; esac
  case "$url" in *\?*) query="?${url#*\?}"; url="${url%%\?*}" ;; esac
  base="${url%/*}"   # scheme://user:pass@host:port
  dbname="${url##*/}"
  case "$base" in *"://"*) : ;; *) die "DB_URL has no database name: '$1'" ;; esac
  [ -n "$dbname" ] || die "DB_URL has no database name: '$1'"
  printf '%s/postgres%s' "$base" "$query"
}

connect_err=""
if ! connect_err="$(psql "$DB_URL_ARG" -tAqc 'SELECT 1' 2>&1)"; then
  case "$connect_err" in
    *"does not exist"*)
      if [ "$CREATE_DB" = "1" ]; then
        log "creating the database named in DB_URL"
        psql "$(maintenance_url "$DB_URL_ARG")" -v ON_ERROR_STOP=1 \
          -c "CREATE DATABASE \"$(printf '%s' "$DB_URL_ARG" | sed 's/[?#].*$//; s|.*/||')\"" \
          || die "CREATE DATABASE failed — the DB_URL role needs CREATEDB on this cluster"
      else
        die "database does not exist yet. Create it as the cluster's admin, e.g.
  psql '$(maintenance_url "$DB_URL_ARG")' -c 'CREATE DATABASE novacron'
or re-run with --create-db (needs CREATEDB for the DB_URL role)"
      fi ;;
    *) die "cannot reach postgres with DB_URL: $connect_err" ;;
  esac
fi

log "applying migrations (novacron-migrate -direction up)"
"$BIN_DIR/novacron-migrate" -db "$DB_URL_ARG" -direction up 2>&1 | sed 's/^/  /' \
  || die "migration failed — fix the database and re-run (migrations are re-runnable; golang-migrate records the version)"
log "schema version: $("$BIN_DIR/novacron-migrate" -db "$DB_URL_ARG" -direction version 2>&1 | tail -n 1)"

# --- systemd units ----------------------------------------------------------

log "installing units into $UNIT_DIR"
install -m 0644 -o root -g root "$REPO_ROOT/deploy/systemd/novacron-api-server.service" "$UNIT_DIR/"
install -m 0644 -o root -g root "$REPO_ROOT/deploy/systemd/novacron-fabric-peer.service" "$UNIT_DIR/"
systemctl daemon-reload

if systemctl is-active --quiet "$OTHER_UNIT"; then
  warn "$OTHER_UNIT is also active on this host; both NovaCron roles now share $ENV_FILE — give them separate API_PORT/NOVACRON_NODE_ID values with 'systemctl edit' drop-ins"
fi

if [ "$NO_START" = "1" ]; then
  log "--no-start: not enabling $ROLE_UNIT"
else
  systemctl enable "$ROLE_UNIT" >/dev/null
  systemctl restart "$ROLE_UNIT"
fi

# --- health -----------------------------------------------------------------

if [ "$NO_START" = "0" ]; then
  health_url="http://127.0.0.1:$API_PORT/health"
  for _ in $(seq 1 30); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "$health_url" 2>/dev/null || true)"
    [ "$code" = "200" ] && break
    sleep 1
  done
  if [ "${code:-}" != "200" ]; then
    warn "$health_url did not return 200 (last code: ${code:-none}). Recent logs:"
    journalctl -u "$ROLE_UNIT" -n 20 --no-pager >&2 || true
    die "install finished but $ROLE_UNIT is not healthy"
  fi
  log "health ok: $health_url"
fi

cat <<EOF
$(date -u +%Y-%m-%dT%H:%M:%SZ) install complete
  role          $ROLE ($ROLE_UNIT)
  node id       $NODE_ID
  advertised    $JOIN_ADDR
  binaries      $BIN_DIR/api-server, $BIN_DIR/novacron-migrate
  config        $ENV_FILE (0600 root:root)
  secret        $SECRET_FILE — copy to the next node with 'install -m 0600' and
                reuse it as NOVACRON_MIGRATION_SECRET there, then run
                deploy/scripts/join-cluster.sh
  vm storage    $STORAGE_PATH
EOF