#!/usr/bin/env bash
# Two-node NovaCron fabric acceptance test.
#
# Provisions two real api-servers (separate databases, separate storage, separate
# ports, distinct node ids) on ONE host, puts node B inside a network namespace
# so inter-node traffic genuinely traverses a veth pair (a same-host pair of IPs
# would route over `lo` and nothing would be shaped — `ip route get <peer>` must
# name the veth, and this script asserts that), joins them through the signed
# join protocol, and then asserts the fabric behaviours end to end:
#
#   1. both nodes visible in GET /api/cluster/nodes with a live link profile
#   2. a job dispatched to the peer completes and its stdout is fetched back
#   3. a second transfer to a busy link is QUEUED with an ETA, not started
#   4. a VM migrated cross-node ends up owned by the target node
#
# Everything is real: real api-server processes, real postgres, real qemu, real
# tc shaping. There are no mocks; a missing prerequisite produces an explicit
# SKIP with the reason, never a silent pass.
#
# Usage:   scripts/fabric/two-node-fabric-test.sh
# Env:     FABRIC_SHAPE_MBIT=50   link rate to shape to (0 disables shaping)
#          FABRIC_DELAY_MS=40     delay to add when netem is available
#          FABRIC_REQUIRE_ALL=1   treat SKIPs as failures (used in CI)
#          FABRIC_KEEP=1          leave nodes/netns/databases behind for inspection
#
# Exit codes: 0 = every assertion passed (skips allowed unless FABRIC_REQUIRE_ALL),
#             1 = at least one assertion failed, 2 = harness misconfiguration.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

# --- configuration ----------------------------------------------------------

RUN_ID="$$"
WORK="${TMPDIR:-/tmp}/fabric-test-$RUN_ID"
BIN="$WORK/api-server"
PGHOST_ADDR="${FABRIC_PGHOST:-127.0.0.1}"
PGPORT="${FABRIC_PGPORT:-5432}"
PGUSER="${FABRIC_PGUSER:-postgres}"
PGPASS="${FABRIC_PGPASSWORD:-postgres}"
DB_A="novacron_fab_a_$RUN_ID"
DB_B="novacron_fab_b_$RUN_ID"
PORT_A="${FABRIC_PORT_A:-18190}"
PORT_B="${FABRIC_PORT_B:-18191}"
NS_B="fabtest-b-$RUN_ID"
VETH_H="veth-fh-$RUN_ID"
VETH_N="veth-fn-$RUN_ID"
ADDR_A="10.96.0.1"
ADDR_B="10.96.0.2"
SHAPE_MBIT="${FABRIC_SHAPE_MBIT:-50}"
DELAY_MS="${FABRIC_DELAY_MS:-40}"
REQUIRE_ALL="${FABRIC_REQUIRE_ALL:-0}"
KEEP="${FABRIC_KEEP:-0}"
HEARTBEAT_WAIT="${FABRIC_HEARTBEAT_WAIT:-45}"

AUTH_SECRET="$(openssl rand -hex 32)"
MIG_SECRET="$(openssl rand -hex 32)"
ADMIN_EMAIL="fabric-test@novacron.invalid"
ADMIN_PASS="Fabr1c!Test1"

PASS=0; FAIL=0; SKIP=0
declare -a RESULTS

say()  { printf '%s\n' "$*"; }
head2() { printf '\n=== %s\n' "$*"; }

ok()   { PASS=$((PASS+1)); RESULTS+=("PASS  $1"); printf '  PASS  %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); RESULTS+=("FAIL  $1"); printf '  FAIL  %s\n' "$1"; }
skip() { SKIP=$((SKIP+1)); RESULTS+=("SKIP  $1 ($2)"); printf '  SKIP  %s (%s)\n' "$1" "$2"; }

# json_get <json> <python-expression-on-d>
json_get() {
  python3 -c 'import json,sys
try:
    d=json.loads(sys.stdin.read())
except Exception:
    print(""); sys.exit(0)
try:
    v=eval(sys.argv[1], {"d": d})
except Exception:
    v=""
print("" if v is None else v)' "$1" <<<"$2"
}

psql_do() { PGPASSWORD="$PGPASS" psql -h "$PGHOST_ADDR" -p "$PGPORT" -U "$PGUSER" -v ON_ERROR_STOP=1 "$@"; }
psql_q()  { PGPASSWORD="$PGPASS" psql -h "$PGHOST_ADDR" -p "$PGPORT" -U "$PGUSER" -tAq "$@"; }

api() { # api <node: a|b> <method> <path> [json-body]
  local node="$1" method="$2" path="$3" body="${4:-}" port addr token
  case "$node" in a) port="$PORT_A"; addr="$ADDR_A";; b) port="$PORT_B"; addr="$ADDR_B";; *) return 2;; esac
  token="$(cat "$WORK/token-$node" 2>/dev/null)"
  if [ -n "$body" ]; then
    curl -sS --max-time 30 -X "$method" "http://$addr:$port$path" \
      -H "Authorization: Bearer $token" -H 'Content-Type: application/json' -d "$body"
  else
    curl -sS --max-time 30 -X "$method" "http://$addr:$port$path" \
      -H "Authorization: Bearer $token"
  fi
}

cleanup() {
  local rc=$?
  head2 "cleanup"
  # node-$n.pid only names the immediate child bash forked ($! for node b is
  # `sudo`'s own PID, not the api-server: `sudo -E ip netns exec ... setsid
  # nohup $BIN` detaches into a brand new session, so killing that PID does
  # NOT reach the actual binary or any qemu child it spawned -- observed
  # live as an orphaned node-b api-server + qemu surviving a "clean" run).
  # $WORK is unique per run (contains this script's own PID), so a plain
  # host-side pkill -f on that exact path safely targets only this run's
  # processes regardless of which netns they're in -- /proc is not
  # namespaced by `ip netns`, so this must NOT be wrapped in `ip netns
  # exec` (that scans the same global /proc and adds no isolation, it only
  # invites running the pattern match somewhere the operator didn't intend).
  for n in a b; do
    [ -f "$WORK/node-$n.pid" ] && sudo kill "$(cat "$WORK/node-$n.pid")" 2>/dev/null
  done
  sudo pkill -TERM -f "$WORK" 2>/dev/null
  sleep 1
  sudo pkill -KILL -f "$WORK" 2>/dev/null
  if [ "$KEEP" = "1" ]; then
    say "  FABRIC_KEEP=1 — leaving nodes, netns ($NS_B), veth pair and databases ($DB_A, $DB_B) in place"
    say "  logs: $WORK"
    return $rc
  fi
  sudo ip netns del "$NS_B" 2>/dev/null
  sudo ip link del "$VETH_H" 2>/dev/null
  psql_do -c "DROP DATABASE IF EXISTS \"$DB_A\"" >/dev/null 2>&1
  psql_do -c "DROP DATABASE IF EXISTS \"$DB_B\"" >/dev/null 2>&1
  sudo rm -rf "$WORK"
  say "  nodes stopped, netns/veth removed, test databases dropped"
  return $rc
}
trap cleanup EXIT INT TERM

# --- prerequisites ----------------------------------------------------------

head2 "prerequisites"
PREREQ_FAIL=""
command -v go >/dev/null      || PREREQ_FAIL="go toolchain"
command -v curl >/dev/null    || PREREQ_FAIL="curl"
command -v python3 >/dev/null || PREREQ_FAIL="python3"
command -v psql >/dev/null    || PREREQ_FAIL="psql client"
command -v ip >/dev/null      || PREREQ_FAIL="iproute2"
command -v tc >/dev/null      || PREREQ_FAIL="tc"
[ -z "$PREREQ_FAIL" ] || { say "missing prerequisite: $PREREQ_FAIL"; exit 2; }
sudo -n true 2>/dev/null || { say "passwordless sudo required (netns + tc)"; exit 2; }
psql_q -c 'SELECT 1' >/dev/null 2>&1 || { say "postgres not reachable at $PGHOST_ADDR:$PGPORT as $PGUSER"; exit 2; }
say "  toolchain, sudo, postgres: ok"

QEMU_BIN="qemu-system-$(uname -m | sed 's/x86_64/x86_64/;s/aarch64/aarch64/')"
HAVE_QEMU=1
if command -v "$QEMU_BIN" >/dev/null || command -v qemu-system-x86_64 >/dev/null || command -v qemu-system-aarch64 >/dev/null; then
  say "  qemu: ok ($QEMU_BIN)"
else
  HAVE_QEMU=0
  say "  qemu: MISSING — migration assertions will be skipped"
fi
command -v socat >/dev/null || PREREQ_FAIL="socat (required: node B in the netns can only reach host postgres through it)"
[ -z "$PREREQ_FAIL" ] || { say "missing prerequisite: $PREREQ_FAIL"; exit 2; }
say "  socat: ok"

mkdir -p "$WORK"

# --- build + databases ------------------------------------------------------

head2 "build"
if ! CGO_ENABLED=1 go build -o "$BIN" ./backend/cmd/api-server; then
  say "api-server build FAILED"; exit 2
fi
say "  built $BIN"

head2 "databases"
for db in "$DB_A" "$DB_B"; do
  psql_do -c "DROP DATABASE IF EXISTS \"$db\"" >/dev/null
  psql_do -c "CREATE DATABASE \"$db\"" >/dev/null || exit 2
done
for db in "$DB_A" "$DB_B"; do
  ( cd database && DB_URL="postgres://$PGUSER:$PGPASS@$PGHOST_ADDR:$PGPORT/$db?sslmode=disable" \
      bash ./scripts/migrate.sh up >"$WORK/migrate-$db.log" 2>&1 ) || {
    say "migrations failed for $db (see $WORK/migrate-$db.log)"; exit 2; }
done
say "  $DB_A and $DB_B created and migrated"

# --- network: node B inside a netns behind a veth ---------------------------

head2 "network"
sudo ip netns del "$NS_B" 2>/dev/null
sudo ip link del "$VETH_H" 2>/dev/null
sudo ip link add "$VETH_H" type veth peer name "$VETH_N" || exit 2
sudo ip addr add "$ADDR_A/24" dev "$VETH_H"
sudo ip link set "$VETH_H" up
sudo ip netns add "$NS_B"
sudo ip link set "$VETH_N" netns "$NS_B"
sudo ip netns exec "$NS_B" ip addr add "$ADDR_B/24" dev "$VETH_N"
sudo ip netns exec "$NS_B" ip link set "$VETH_N" up
sudo ip netns exec "$NS_B" ip link set lo up
say "  $VETH_H=$ADDR_A <-> $NS_B:$VETH_N=$ADDR_B"

ROUTE_DEV="$(ip route get "$ADDR_B" | head -1 | sed -n 's/.*dev \([^ ]*\).*/\1/p')"
if [ "$ROUTE_DEV" = "$VETH_H" ]; then
  ok "inter-node path traverses the veth (ip route get $ADDR_B -> dev $ROUTE_DEV)"
else
  bad "inter-node path does NOT traverse the veth (ip route get $ADDR_B -> dev ${ROUTE_DEV:-?}); traffic would bypass shaping"
fi

SHAPING_MODE="none"
if [ "$SHAPE_MBIT" != "0" ]; then
  if sudo ip netns exec "$NS_B" tc qdisc add dev "$VETH_N" root netem rate "${SHAPE_MBIT}mbit" delay "${DELAY_MS}ms" 2>/dev/null \
     && sudo tc qdisc add dev "$VETH_H" root netem rate "${SHAPE_MBIT}mbit" delay "${DELAY_MS}ms" 2>/dev/null; then
    SHAPING_MODE="netem ${SHAPE_MBIT}mbit/${DELAY_MS}ms"
  else
    sudo tc qdisc del dev "$VETH_H" root 2>/dev/null
    sudo ip netns exec "$NS_B" tc qdisc del dev "$VETH_N" root 2>/dev/null
    if sudo tc qdisc add dev "$VETH_H" root tbf rate "${SHAPE_MBIT}mbit" burst 32kbit latency 400ms 2>/dev/null \
       && sudo ip netns exec "$NS_B" tc qdisc add dev "$VETH_N" root tbf rate "${SHAPE_MBIT}mbit" burst 32kbit latency 400ms 2>/dev/null; then
      SHAPING_MODE="tbf ${SHAPE_MBIT}mbit (netem unavailable on this kernel)"
    else
      SHAPING_MODE="none (tc shaping unavailable)"
    fi
  fi
fi
say "  shaping: $SHAPING_MODE"
[ "$SHAPING_MODE" = "none" ] && skip "link shaping" "tc could not shape the veth; timings will be LAN-speed"

# --- start the two nodes ----------------------------------------------------

start_node() { # start_node <a|b>
  local n="$1" port storage db_url host_env
  if [ "$n" = "a" ]; then
    port="$PORT_A"; storage="$WORK/storage-a"; db_url="postgres://$PGUSER:$PGPASS@$PGHOST_ADDR:$PGPORT/$DB_A?sslmode=disable"
    host_env=""
  else
    port="$PORT_B"; storage="$WORK/storage-b"; db_url="postgres://$PGUSER:$PGPASS@$ADDR_A:$PORT_SOCAT_B/$DB_B?sslmode=disable"
    host_env="API_HOST=$ADDR_B"
  fi
  mkdir -p "$storage"
  local env_common=(
    LOG_LEVEL=info
    AUTH_SECRET="$AUTH_SECRET"
    NOVACRON_MIGRATION_SECRET="$MIG_SECRET"
    DB_URL="$db_url"
    STORAGE_PATH="$storage"
    API_PORT="$port"
    WS_PORT="$((port+1000))"
    NOVACRON_NODE_ID="fab-$n"
    NOVACRON_PROBE_BYTES=2097152
  )

  local join_addr
  if [ "$n" = "a" ]; then
    join_addr="$ADDR_A"
  else
    join_addr="$ADDR_B"
  fi
  [ -n "$host_env" ] && env_common+=("$host_env")
  [ "$n" = "b" ] && env_common+=("NOVACRON_JOIN_PEERS=$ADDR_A:$PORT_A")
  [ -n "${REDIS_URL_AVAILABLE:-}" ] && env_common+=("REDIS_URL=$REDIS_URL_AVAILABLE")
  env_common+=("NOVACRON_JOIN_ADDR=$join_addr:$port")

  if [ "$n" = "a" ]; then
    env "${env_common[@]}" setsid nohup "$BIN" >"$WORK/node-a.log" 2>&1 < /dev/null &
  else
    sudo -E ip netns exec "$NS_B" env "${env_common[@]}" setsid nohup "$BIN" >"$WORK/node-b.log" 2>&1 < /dev/null &
  fi
  echo $! > "$WORK/node-$n.pid"
}

# node B's postgres/redis are only reachable through the host-side veth IP, so
# forward those two ports there (postgres listens on 127.0.0.1 of the host).
PORT_SOCAT_B="$((PGPORT+10000))"
socat TCP-LISTEN:"$PORT_SOCAT_B",bind="$ADDR_A",fork,reuseaddr TCP:127.0.0.1:"$PGPORT" >"$WORK/socat-pg.log" 2>&1 &
echo $! > "$WORK/socat-pg.pid"
if command -v redis-cli >/dev/null 2>&1; then
  REDIS_FWD="$((PGPORT+10001))"
  REDIS_URL_AVAILABLE="redis://$ADDR_A:$REDIS_FWD"
  socat TCP-LISTEN:"$REDIS_FWD",bind="$ADDR_A",fork,reuseaddr TCP:127.0.0.1:6379 >"$WORK/socat-redis.log" 2>&1 &
  echo $! > "$WORK/socat-redis.pid"
fi
sleep 1

head2 "start nodes"
start_node a
start_node b

wait_health() { # wait_health <addr:port> <seconds>
  local target="$1" deadline=$(( $(date +%s) + $2 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$target/health" 2>/dev/null)" = "200" ]; then return 0; fi
    sleep 1
  done
  return 1
}

HEALTHY=1
if wait_health "$ADDR_A:$PORT_A" 30; then ok "node A healthy on $ADDR_A:$PORT_A"; else bad "node A did not become healthy"; HEALTHY=0; fi
if wait_health "$ADDR_B:$PORT_B" 30; then ok "node B healthy on $ADDR_B:$PORT_B (inside $NS_B)"; else bad "node B did not become healthy"; HEALTHY=0; fi
if [ "$HEALTHY" = "0" ]; then
  say "--- node A log tail"; tail -5 "$WORK/node-a.log" 2>/dev/null | cut -c1-200
  say "--- node B log tail"; tail -5 "$WORK/node-b.log" 2>/dev/null | cut -c1-200
  say "RESULT: harness could not start both nodes"; exit 1
fi

# --- auth -------------------------------------------------------------------

for n in a b; do
  port="$PORT_A"; addr="$ADDR_A"; [ "$n" = "b" ] && { port="$PORT_B"; addr="$ADDR_B"; }
  curl -sS --max-time 20 -X POST "http://$addr:$port/api/auth/register" -H 'Content-Type: application/json' \
    -d "{\"email\":\"$ADMIN_EMAIL\",\"password\":\"$ADMIN_PASS\",\"username\":\"fabtest\",\"name\":\"Fabric Test\"}" >/dev/null
  login_resp="$(curl -sS --max-time 20 -X POST "http://$addr:$port/api/auth/login" -H 'Content-Type: application/json' \
        -d "{\"email\":\"$ADMIN_EMAIL\",\"password\":\"$ADMIN_PASS\"}")"
  tok="$(json_get 'd.get("token","")' "$login_resp")"
  if [ -z "$tok" ]; then bad "login on node $n"; else echo "$tok" > "$WORK/token-$n"; fi
done
[ -s "$WORK/token-a" ] && [ -s "$WORK/token-b" ] && ok "operator account usable on both nodes"

# --- assertion 1: membership + link profile ---------------------------------

head2 "assertion 1: membership and link profile"
JOINED=0
deadline=$(( $(date +%s) + HEARTBEAT_WAIT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  nodes_json="$(api a GET /api/cluster/nodes)"
  reach="$(json_get '[n.get("reachable") for n in d.get("nodes",[]) if n.get("node_id")=="fab-b"][0] if [n for n in d.get("nodes",[]) if n.get("node_id")=="fab-b"] else False' "$nodes_json")"
  [ "$reach" = "True" ] && { JOINED=1; break; }
  sleep 3
done
if [ "$JOINED" = "1" ]; then
  ok "node B joined and is reachable from A (GET /api/cluster/nodes)"
  # throughput is populated by the heartbeat loop's ticker (fires every
  # heartbeatInterval=30s server-side, not immediately on join), so this
  # needs its own poll window separate from the reachability wait above.
  tp=0; rtt=0
  tp_deadline=$(( $(date +%s) + HEARTBEAT_WAIT ))
  while [ "$(date +%s)" -lt "$tp_deadline" ]; do
    links_json="$(api a GET /api/cluster/links)"
    tp="$(json_get '[(l.get("throughput_bps") or 0) for l in d.get("links",[]) if l.get("node_id")=="fab-b"][0] if [l for l in d.get("links",[]) if l.get("node_id")=="fab-b"] else 0' "$links_json")"
    rtt="$(json_get '[(l.get("rtt_ms") or 0) for l in d.get("links",[]) if l.get("node_id")=="fab-b"][0] if [l for l in d.get("links",[]) if l.get("node_id")=="fab-b"] else 0' "$links_json")"
    python3 -c "import sys; sys.exit(0 if float('${tp:-0}')>0 else 1)" && break
    sleep 3
  done
  tp_mbps="$(python3 -c "print(f'{float('$tp')/1e6:.1f}')" 2>/dev/null || echo 0)"
  say "    measured link: ${tp_mbps} Mbps, rtt ${rtt} ms (shaping: $SHAPING_MODE)"
  if python3 -c "import sys; sys.exit(0 if float('${tp:-0}')>0 else 1)"; then
    ok "link profile carries a measured throughput (${tp_mbps} Mbps)"
  else
    bad "link profile has no measured throughput after ${HEARTBEAT_WAIT}s"
  fi
else
  bad "node B never became reachable from A"
fi

# --- assertion 2: job dispatched to the peer --------------------------------

head2 "assertion 2: job dispatched to the peer"
JOB_MARKER="fabric-job-$RUN_ID"
job_json="$(api a POST /api/compute/jobs "{\"name\":\"acceptance-$RUN_ID\",\"command\":\"sh\",\"args\":[\"-c\",\"echo $JOB_MARKER; hostname\"],\"node_id\":\"fab-b\"}")"
JOB_ID="$(json_get 'd.get("job_id","")' "$job_json")"
JOB_NODE="$(json_get 'd.get("node_id","")' "$job_json")"
if [ -n "$JOB_ID" ] && [ "$JOB_NODE" = "fab-b" ]; then
  ok "job $JOB_ID placed on the peer (node_id=fab-b)"
  JOB_OK=0
  deadline=$(( $(date +%s) + 60 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    detail="$(api a GET "/api/compute/jobs/$JOB_ID")"
    status="$(json_get 'd.get("status","")' "$detail")"
    stdout="$(json_get '(d.get("logs") or {}).get("stdout","")' "$detail")"
    case "$status" in
      completed) if printf '%s' "$stdout" | grep -q "$JOB_MARKER"; then JOB_OK=1; fi; break;;
      failed) say "    job failed: $(json_get 'd.get("error","")' "$detail")"; break;;
    esac
    sleep 3
  done
  if [ "$JOB_OK" = "1" ]; then
    ok "job completed on the peer and its stdout was fetched back through node A"
  else
    bad "job did not complete with the expected stdout (status=${status:-?})"
  fi
else
  bad "job submission did not place the job on the peer (response: $job_json)"
fi

# --- assertion 3: transfer admission queue ----------------------------------

head2 "assertion 3: second transfer to a busy link is queued with an ETA"
QUEUE_OK=0; QUEUE_MSG=""
if [ "$HAVE_QEMU" = "0" ]; then
  skip "transfer queue" "qemu not installed, cannot create a migratable VM"
else
  IMG="$WORK/cirros.img"
  CIRROS_URL="${FABRIC_CIRROS_URL:-}"
  if [ -z "$CIRROS_URL" ]; then
    case "$(uname -m)" in
      aarch64|arm64) CIRROS_URL="https://download.cirros-cloud.net/0.6.2/cirros-0.6.2-aarch64-disk.img";;
      *)             CIRROS_URL="https://download.cirros-cloud.net/0.6.2/cirros-0.6.2-x86_64-disk.img";;
    esac
  fi
  if curl -sSL --max-time 300 -o "$IMG" "$CIRROS_URL" && [ -s "$IMG" ]; then
    vm_json="$(api a POST /api/vms "{\"name\":\"queue-vm-$RUN_ID\",\"image\":\"$IMG\",\"memory_mb\":256,\"vcpus\":1,\"disk_size_gb\":1,\"node_id\":\"fab-a\"}")"
    VM_ID="$(json_get 'd.get("id","")' "$vm_json")"
    if [ -n "$VM_ID" ]; then
      api a POST "/api/vms/$VM_ID/start" >/dev/null
      sleep 5
      t1="$(api a POST /api/transfers "{\"kind\":\"migration\",\"vm_id\":\"$VM_ID\",\"target_node\":\"fab-b\",\"migration_type\":\"block\"}")"
      t2="$(api a POST /api/transfers "{\"kind\":\"migration\",\"vm_id\":\"$VM_ID\",\"target_node\":\"fab-b\",\"migration_type\":\"block\"}")"
      T1_ID="$(json_get 'd.get("transfer_id","")' "$t1")"
      s1="$(json_get 'd.get("status","")' "$t1")"
      s2="$(json_get 'd.get("status","")' "$t2")"
      eta2="$(json_get 'd.get("eta_seconds")' "$t2")"
      if [ "$s1" = "running" ] && [ "$s2" = "queued" ] && [ -n "$eta2" ] && [ "$eta2" != "None" ]; then
        QUEUE_OK=1
        QUEUE_MSG="T1 running, T2 queued with eta_seconds=$eta2"
      else
        QUEUE_MSG="T1=$s1 T2=$s2 eta=$eta2 (first transfer may have finished before the second was submitted)"
      fi
    else
      QUEUE_MSG="VM creation failed: $vm_json"
    fi
  else
    QUEUE_MSG="cirros image download failed"
  fi
  if [ "$QUEUE_OK" = "1" ]; then ok "transfer admission: $QUEUE_MSG"; else bad "transfer admission: $QUEUE_MSG"; fi
fi

# --- assertion 4: migration lands the VM on the target ----------------------

head2 "assertion 4: cross-node migration moves the VM to the target"
MIG_OK=0; MIG_MSG=""
if [ "$HAVE_QEMU" = "0" ]; then
  skip "cross-node migration" "qemu not installed"
elif [ -z "${VM_ID:-}" ] || [ -z "${T1_ID:-}" ]; then
  skip "cross-node migration" "no VM/transfer available (see the transfer assertion above)"
else
  # Track T1 specifically by its own transfer_id: assertion 3 admits a second,
  # redundant migration (T2) for the SAME vm_id to prove the admission queue,
  # so polling /api/transfers by vm_id is ambiguous between the two records.
  # T2 legitimately fails once T1 has already moved the VM off this node.
  deadline=$(( $(date +%s) + 600 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    t1_detail="$(api a GET "/api/transfers/$T1_ID")"
    st="$(json_get 'd.get("status","")' "$t1_detail")"
    case "$st" in
      completed) MIG_OK=1; MIG_MSG="transfer completed"; break;;
      failed) MIG_MSG="transfer failed: $(json_get 'd.get("error","")' "$t1_detail")"; break;;
    esac
    sleep 5
  done
  if [ "$MIG_OK" = "1" ]; then
    b_view="$(api b GET "/api/vms/$VM_ID")"
    b_state="$(json_get 'd.get("state") or d.get("error") or ""' "$b_view")"
    if [ "$b_state" = "running" ]; then
      ok "VM $VM_ID is running on the target node (GET /api/vms/$VM_ID on node B)"
    else
      bad "transfer completed but node B reports the VM as '${b_state:-?}'"
    fi
  else
    bad "cross-node migration did not complete: ${MIG_MSG:-timeout}"
  fi
fi

# --- summary ----------------------------------------------------------------

head2 "summary"
for line in "${RESULTS[@]}"; do say "  $line"; done
say ""
say "  shaping used: $SHAPING_MODE"
say "  pass=$PASS fail=$FAIL skip=$SKIP"
if [ "$FAIL" -gt 0 ]; then
  say "RESULT: FAIL"
  exit 1
fi
if [ "$REQUIRE_ALL" = "1" ] && [ "$SKIP" -gt 0 ]; then
  say "RESULT: FAIL (FABRIC_REQUIRE_ALL=1 and $SKIP assertion(s) were skipped)"
  exit 1
fi
say "RESULT: PASS${SKIP:+ ($SKIP skipped)}"
exit 0
