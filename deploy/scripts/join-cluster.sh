#!/usr/bin/env bash
# Join this node to an existing NovaCron fabric with the signed join protocol.
#
# The seed node accepts POST /internal/cluster/join only when the request
# carries an HMAC-SHA256 signature over "node_id|addr|ts" keyed by the fabric's
# shared secret (NOVACRON_MIGRATION_SECRET), the timestamp is within 60s, and
# the joining node answers a callback probe on the addr it advertised — a node
# that is not actually up cannot join. This script speaks that protocol, then
# pins the seed into NOVACRON_JOIN_PEERS so the node rejoins by itself after a
# reboot (the in-process join is what persists the seed in this node's own
# cluster_peers table).
#
# The shared secret must be byte-identical on both nodes: install-node.sh
# generates it on the first node (/etc/novacron/migration.secret) and it is
# copied to every node afterwards.
#
# Usage:  sudo deploy/scripts/join-cluster.sh --seed HOST:PORT [options]
# Exit:   0 = joined (and verified), 1 = failed, 2 = usage error.

set -euo pipefail

ENV_FILE="/etc/novacron/novacron.env"
SECRET_FILE="/etc/novacron/migration.secret"
SEEDS=""
UNIT=""
TIMEOUT=15
NO_RESTART=0
DRY_RUN=0

log()  { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die()  { printf '%s ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 1; }

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --seed HOST:PORT    seed node's API addr (comma-separated list allowed)
  --env-file PATH     node environment file (default /etc/novacron/novacron.env)
  --secret-file PATH  fabric shared secret (default /etc/novacron/migration.secret)
  --unit NAME         unit to restart after joining (default: whichever NovaCron
                      unit is active, else novacron-api-server.service)
  --timeout SECONDS   per-seed request timeout (default 15)
  --no-restart        do not restart the unit (the join still registers this
                      node; NOVACRON_JOIN_PEERS is still persisted)
  --dry-run           print the payload and signature without POSTing
  -h, --help          this text

Root is required only when using the default env file (0600 root:root) or when
the unit is restarted; --env-file plus --no-restart runs unprivileged.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --seed)        SEEDS="${2:?--seed needs a value}"; shift 2 ;;
    --env-file)    ENV_FILE="${2:?--env-file needs a value}"; shift 2 ;;
    --secret-file) SECRET_FILE="${2:?--secret-file needs a value}"; shift 2 ;;
    --unit)        UNIT="${2:?--unit needs a value}"; shift 2 ;;
    --timeout)     TIMEOUT="${2:?--timeout needs a value}"; shift 2 ;;
    --no-restart)  NO_RESTART=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             usage >&2; die "unknown argument: $1" ;;
  esac
done

[ -n "$SEEDS" ] || { usage >&2; die "--seed is required (the addr of a node already in the fabric)"; }
[ -r "$ENV_FILE" ] || die "cannot read $ENV_FILE — run install-node.sh on this node first, or pass --env-file"
case "$TIMEOUT" in ''|*[!0-9]*) die "--timeout must be a number of seconds, got '$TIMEOUT'";; esac

if [ "$(id -u)" != "0" ]; then
  if [ "$ENV_FILE" = "/etc/novacron/novacron.env" ]; then
    die "must run as root to read the 0600 env file: sudo $0 $*"
  fi
  if [ "$NO_RESTART" = "0" ]; then
    die "restarting a unit needs root: re-run with sudo, or pass --no-restart"
  fi
fi

for bin in curl openssl awk; do
  command -v "$bin" >/dev/null 2>&1 || die "missing '$bin' — install-node.sh lists the per-binary install commands"
done

env_get() { sed -n "s/^$1=//p" "$ENV_FILE" | tail -n 1 | sed 's/^"\(.*\)"$/\1/'; }

SECRET="$(env_get NOVACRON_MIGRATION_SECRET)"
NODE_ID="$(env_get NOVACRON_NODE_ID)"
JOIN_ADDR="$(env_get NOVACRON_JOIN_ADDR)"
[ -n "$SECRET" ] || die "NOVACRON_MIGRATION_SECRET is not set in $ENV_FILE — the join would be rejected (the protocol fails closed without a secret)"
[ -n "$NODE_ID" ] || die "NOVACRON_NODE_ID is not set in $ENV_FILE"
if [ "$NODE_ID" = "local" ]; then
  die "NOVACRON_NODE_ID is 'local', the api-server's unconfigured default: every such node collides in the seed's peer map. Set a unique id in $ENV_FILE."
fi
if [ -z "$JOIN_ADDR" ]; then
  # Same fallback the api-server uses (selfJoinAddr): API_HOST:API_PORT.
  host="$(env_get API_HOST)"; [ -n "$host" ] || host="127.0.0.1"
  port="$(env_get API_PORT)"; [ -n "$port" ] || port="8090"
  JOIN_ADDR="$host:$port"
  log "WARNING: NOVACRON_JOIN_ADDR is not set; advertising $JOIN_ADDR. Peers must be able to reach that addr or the join is rejected."
fi

# The env file is what the unit loads; migration.secret is what operators copy
# between nodes. A mismatch means the seed will reject every request with a
# signature error that looks like a protocol bug, so catch it here.
if [ -r "$SECRET_FILE" ]; then
  FILE_SECRET="$(tr -d '\r\n' < "$SECRET_FILE")"
  if [ -n "$FILE_SECRET" ] && [ "$FILE_SECRET" != "$SECRET" ]; then
    die "$SECRET_FILE and NOVACRON_MIGRATION_SECRET in $ENV_FILE differ. Make them identical (the secret in $ENV_FILE is the one the running unit uses), e.g.
  printf '%s\\n' \"\$(sed -n 's/^NOVACRON_MIGRATION_SECRET=//p' $ENV_FILE | tail -n1)\" > $SECRET_FILE
then distribute that file to the other nodes."
  fi
fi

sign() { # sign <node_id> <addr> <ts> — must byte-match the Go side:
         # hmac.New(sha256, secret) over fmt.Sprintf("%s|%s|%d", nodeID, addr, ts)
  printf '%s|%s|%s' "$1" "$2" "$3" \
    | openssl dgst -sha256 -hmac "$SECRET" -r | awk '{print $1}'
}

TS="$(date +%s)"
SIG="$(sign "$NODE_ID" "$JOIN_ADDR" "$TS")"
PAYLOAD="{\"node_id\":\"$NODE_ID\",\"addr\":\"$JOIN_ADDR\",\"ts\":$TS}"

if [ "$DRY_RUN" = "1" ]; then
  log "dry-run: node_id=$NODE_ID addr=$JOIN_ADDR ts=$TS"
  log "dry-run: X-Join-Signature: $SIG"
  log "dry-run: POST http://<seed>/internal/cluster/join -d '$PAYLOAD'"
  exit 0
fi

JOINED_ANY=0
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
for seed in "${SEED_LIST[@]}"; do
  [ -n "$seed" ] || continue
  # Re-sign per seed so the ts stays inside the 60s freshness window even when
  # several seeds are listed.
  TS="$(date +%s)"
  SIG="$(sign "$NODE_ID" "$JOIN_ADDR" "$TS")"
  PAYLOAD="{\"node_id\":\"$NODE_ID\",\"addr\":\"$JOIN_ADDR\",\"ts\":$TS}"

  log "joining $seed as $NODE_ID ($JOIN_ADDR)"
  body=""
  if ! body="$(curl -sS --max-time "$TIMEOUT" -X POST "http://$seed/internal/cluster/join" \
      -H 'Content-Type: application/json' \
      -H "X-Join-Signature: $SIG" \
      -d "$PAYLOAD" 2>&1)"; then
    # Connection refused / timeout here usually means the seed's API is down or
    # its port is blocked; the joiner's own reachability is checked by the seed.
    log "  FAILED to reach $seed: $body"
    continue
  fi

  if ! printf '%s' "$body" | grep -q '"joined":true'; then
    log "  REJECTED by $seed: $body"
    continue
  fi
  # The seed registers the joiner before answering, so its peer list must
  # already contain us — that is the proof this join took effect.
  if ! printf '%s' "$body" | grep -q "\"node_id\":\"$NODE_ID\""; then
    log "  UNCONFIRMED: $seed answered without $NODE_ID in its peer list: $body"
    continue
  fi

  log "  joined: $seed registered $NODE_ID"
  JOINED_ANY=1
done

[ "$JOINED_ANY" = "1" ] || die "no seed accepted the join (see the per-seed output above). Check: the seed's API is reachable on that port, both nodes carry the same NOVACRON_MIGRATION_SECRET, this node's api-server is running and reachable at $JOIN_ADDR, and the clocks agree within 60s."

# Persist the seeds so the node rejoins at boot with its own in-process join
# (that is the path that writes the seed row into this node's cluster_peers).
if [ "$(env_get NOVACRON_JOIN_PEERS)" = "$SEEDS" ]; then
  log "NOVACRON_JOIN_PEERS already set to $SEEDS"
else
  TMP="$(mktemp)"
  awk -v peers="$SEEDS" '
    /^NOVACRON_JOIN_PEERS=/ { print "NOVACRON_JOIN_PEERS=" peers; seen = 1; next }
    { print }
    END { if (!seen) print "NOVACRON_JOIN_PEERS=" peers }
  ' "$ENV_FILE" > "$TMP"
  install -m 0600 -o root -g root "$TMP" "$ENV_FILE" 2>/dev/null \
    || install -m 0600 "$TMP" "$ENV_FILE"
  rm -f "$TMP"
  log "persisted NOVACRON_JOIN_PEERS=$SEEDS in $ENV_FILE"
fi

if [ -z "$UNIT" ]; then
  if command -v systemctl >/dev/null 2>&1; then
    for candidate in novacron-fabric-peer.service novacron-api-server.service; do
      if systemctl is-active --quiet "$candidate" 2>/dev/null; then UNIT="$candidate"; break; fi
    done
  fi
  [ -n "$UNIT" ] || UNIT="novacron-api-server.service"
fi

if [ "$NO_RESTART" = "1" ]; then
  log "--no-restart: restart the unit yourself so the node rejoins at boot: sudo systemctl restart $UNIT"
  exit 0
fi

log "restarting $UNIT so the node rejoins (and persists the seed) at boot"
systemctl restart "$UNIT"

log "joined: the node will rejoin by itself after a reboot"
cat <<'EOF'
Verify from the seed (operator JWT required):
  TOKEN=$(curl -sS -X POST http://localhost:8090/api/auth/login \
            -H 'Content-Type: application/json' \
            -d '{"email":"you@example.com","password":"..."}' \
          | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
  curl -sS http://<seed>:<port>/api/cluster/links -H "Authorization: Bearer $TOKEN"
Measured link RTT/throughput appear within the heartbeat interval (~30s).
EOF