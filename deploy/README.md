# Deploying NovaCron on real hardware

This directory is the operator surface: two systemd units, the install/join
scripts, database backup/restore, and the soak canary. Everything here assumes
Debian/Ubuntu (apt names) on a host you control with sudo.

Both units run the **same binary** — `./backend/cmd/api-server` — because that
binary serves both roles in this repo: it is the HTTP API *and* the fabric peer
(join/heartbeat, compute jobs, transfers, VM migration). The units differ in the
role they configure:

| Unit | Role | Port | Use |
|---|---|---|---|
| `novacron-api-server.service` | control plane | 8090 (`WS_PORT` 8091) | API/UI + fabric seed; accepts joins |
| `novacron-fabric-peer.service` | compute node | 9000 (`WS_PORT` 9001) | joins a seed, runs jobs and VMs |

Run **one role per host**. `deploy/systemd/novacron-api-server.service` is the
canonical control-plane unit; `systemd/novacron-api.service` at the repo root is
a symlink to it, kept so existing references resolve to the corrected,
KVM-capable unit.

## 1. What you must provision

| Requirement | Where | Notes |
|---|---|---|
| postgres >= 13 | at least one per fabric | Migrations create the schema and the `uuid-ossp`/`pgcrypto` extensions themselves. |
| DB role + database | e.g. `CREATE ROLE novacron LOGIN PASSWORD '...'; CREATE DATABASE novacron OWNER novacron;` | `install-node.sh --create-db` can create the database, never the role. |
| `qemu-system-<arch>` + `qemu-img` | compute nodes | Guests are launched directly; `qemu-system-aarch64` on arm64, `-x86_64` on x86_64. |
| `/dev/kvm` | nodes that run VMs | Device access, not a capability — the units add the `novacron` user to the `kvm` group. Without it guests fall back to TCG (very slow) and the install warns. |
| `psql` client | every node | migrations, `backup-db.sh`, `restore-db.sh`. |
| `socat`, `tc` (`iproute2`), `curl`, `openssl` | every node | `tc`/`socat` are what the acceptance harness shapes and routes with; `openssl` signs joins and generates secrets; `curl` carries health and join requests. |
| Go toolchain | install host (unless `--binary`/`--skip-build`) | `install-node.sh` builds `api-server` and the embedded-migration `novacron-migrate` from source. |
| Two port ranges per node | firewall | the API port (peers and users must reach it) **and** an ephemeral TCP port per migration: the destination allocates a free local port for `-incoming`, so either allow the node's ephemeral range or keep migrations on a trusted network. |
| kernel netem / tbf | optional | only the harness needs shaping. netem is not required: it falls back to `tbf`, and if neither exists the run reports a `SKIP` instead of pretending to measure a shaped link. |

No libvirt, no docker, no redis, no external `migrate` CLI are required.

## 2. Paths, ports, configuration

| Path | Contents |
|---|---|
| `/opt/novacron/bin/api-server` | the service binary (path matches `apparmor/novacron-api`) |
| `/opt/novacron/bin/novacron-migrate` | migrator built from `./database`, migrations embedded |
| `/etc/novacron/novacron.env` | **all** service environment, mode `0600 root:root`; holds `DB_URL`, `AUTH_SECRET`, `NOVACRON_MIGRATION_SECRET` |
| `/etc/novacron/migration.secret` | the fabric shared secret alone, so it can be copied to the next node |
| `/var/lib/novacron/{vms,volumes}` | VM disks (`vms/<id>/disk.qcow2` + `qemu.pid`) and volumes — the only node-local state that is not in postgres |
| `/var/log/novacron` | used only when `LOG_OUTPUT` names a file; by default logs go to journald |

Recognised environment (the env file wins over the unit's `Environment=`
defaults): `DB_URL`, `AUTH_SECRET`, `NOVACRON_MIGRATION_SECRET`,
`NOVACRON_NODE_ID`, `NOVACRON_JOIN_ADDR`, `NOVACRON_JOIN_PEERS`, `API_PORT`,
`WS_PORT`, `STORAGE_PATH`, `LOG_LEVEL`, `LOG_FORMAT`, `LOG_OUTPUT`, plus the
usage-metering rate card (`NOVACRON_RATE_PER_VCPU_HOUR`,
`NOVACRON_RATE_PER_GB_EGRESS`, `NOVACRON_RATE_PER_JOB_SECOND`,
`NOVACRON_RATE_PER_MIGRATION` — default `0` records usage without pricing it).

`AUTH_SECRET` is per node (it signs that node's sessions). Only
`NOVACRON_MIGRATION_SECRET` must be identical across the fabric; it is checked
in constant time by every `/internal/*` handler, and an empty value makes the
node refuse all peer RPCs (fail closed). `NOVACRON_NODE_ID` must be unique per
host and must not be `local` (the api-server's unconfigured default, which
collides in every peer map).

## 3. Install a control-plane node

```bash
sudo deploy/scripts/install-node.sh --role control \
     --db-url 'postgresql://novacron:secret@127.0.0.1:5432/novacron?sslmode=disable' \
     --create-db
```

The script refuses to run as a non-root user, verifies `psql`, `qemu-system-*`,
`qemu-img`, `socat`, `tc`, `openssl`, `curl` (and `go` when building) with a
per-binary install command in the error, then:

1. creates the `novacron` system user (plus `kvm` group membership) and
   `/opt/novacron`, `/var/lib/novacron/{vms,volumes}`, `/var/log/novacron`;
2. generates `/etc/novacron/migration.secret` and `/etc/novacron/novacron.env`
   (32-byte hex secrets via `openssl rand`), mode `0600 root:root`;
3. builds and installs both binaries;
4. applies migrations with the repo's own migrator
   (`novacron-migrate -db ... -direction up`) and prints the resulting schema
   version;
5. installs both units, then enables and starts the role's unit;
6. polls `http://127.0.0.1:<port>/health` for 30s and dumps the last 20 journal
   lines if it never returns 200.

Re-running it is safe: an existing env file is reused (use `--force-env` to
rotate secrets — that invalidates sessions and requires re-distributing
`migration.secret`), and migrations are version-tracked.

Useful flags: `--role compute`, `--node-id`, `--join-addr HOST:PORT`,
`--join-peers`, `--api-port`, `--storage-path`, `--binary PATH`, `--skip-build`,
`--no-start`.

## 4. Join a compute node to the fabric

The join protocol is HMAC-signed and fails closed: `POST
/internal/cluster/join` with `X-Join-Signature = HMAC-SHA256(secret,
"node_id|addr|ts")`, `|now-ts| < 60s`, and the seed then calls the joiner back
at its advertised addr (`/internal/cluster/capacity`, same secret) — an
unreachable or secret-less node is rejected, so a bogus address cannot poison
the peer map.

```bash
# on the new node: same shared secret as the rest of the fabric
sudo install -m 0600 -o root -g root /path/from/seed/migration.secret /etc/novacron/migration.secret
sudo sed -i "s/^NOVACRON_MIGRATION_SECRET=.*/NOVACRON_MIGRATION_SECRET=$(sudo cat /etc/novacron/migration.secret)/" /etc/novacron/novacron.env
sudo systemctl restart novacron-fabric-peer

# then join (verifies the seed registered us, then persists the seed)
sudo deploy/scripts/join-cluster.sh --seed 10.0.0.1:8090
```

`join-cluster.sh` refuses when the node id is `local`, when the secret file and
the env file disagree (an otherwise baffling signature error), or when no seed
accepted the join. On success it writes `NOVACRON_JOIN_PEERS` into the env file
and restarts the unit, so the node rejoins by itself at boot — the in-process
join is what persists the seed in the node's own `cluster_peers` table.
`--dry-run` prints the payload and signature without sending anything;
`--no-restart` plus `--env-file` runs unprivileged.

## 5. Healthcheck

`GET /health` (unauthenticated) is the liveness probe:

```bash
curl -sS http://127.0.0.1:8090/health
# {"status":"healthy","timestamp":"...","version":"1.0.0","service":"novacron-api",
#  "checks":{"database":"ok","storage":"ok"}}
```

`200` + `status:"healthy"` means the database answered within 2s and
`STORAGE_PATH` exists; a failed DB ping returns `503` with
`status:"unhealthy"` and the error text in `checks.database`. A missing
`STORAGE_PATH` only warns (`checks.storage`) — it stays `200` because the
process can still serve control-plane traffic. `status` is `unhealthy` while
`state` is `degraded` in the payload — check `checks` before alerting on
`status` alone. The WebSocket endpoint is served by the same HTTP server on the
API port; `WS_PORT` is only a log line in this build.

Fleet-level checks after a start: `GET /api/cluster/nodes` and
`GET /api/cluster/links` (operator JWT) show per-peer capacity, measured RTT and
throughput; the link profile is refreshed by a heartbeat every ~30s.

## 6. Backups and restore

```bash
sudo deploy/scripts/backup-db.sh                     # -> /var/backups/novacron/<db>-<stamp>.dump (+.sha256, +.meta)
sudo deploy/scripts/backup-db.sh --keep 14 --out /srv/backups

sudo systemctl stop novacron-api-server
sudo deploy/scripts/restore-db.sh --from /var/backups/novacron/novacron-20260922T101500Z.dump
sudo systemctl start novacron-api-server
```

Both scripts read `DB_URL` from `--db-url`, `$DB_URL`, or the env file (in that
order), so they always target the database the units use. `backup-db.sh` writes
pg_dump's custom format (checksummed, and re-read with `pg_restore --list` to
prove it is restorable) and records the golang-migrate schema version in the
`.meta` file. `restore-db.sh` verifies the checksum, refuses to run while a
NovaCron unit is active or while the target database still has tables (unless
`--force`), and restores in a single transaction so a failure leaves the
previous state intact.

**VM disks are not in that backup.** `backup-db.sh` covers the control-plane
database only; `<STORAGE_PATH>` (`/var/lib/novacron/vms`) needs its own
snapshotting (LVM/ZFS/pg-free host snapshots) if you want to survive a disk
failure on a compute node. Nothing in this repo replicates guest disks.

## 7. Node-loss drill

There is **no automatic failover of a guest today**. What actually happens when
a node dies with a VM on it:

1. **Detection.** The dead node's heartbeat stops; within ~30s the surviving
   nodes report it `reachable:false` in `GET /api/cluster/nodes` and its
   `link.measured_at` goes stale. Its postgres rows do not change.
2. **Control-plane state.** The VM row still says `state='running'`,
   `node_id='<dead node>'`, and the disk it depends on
   (`/var/lib/novacron/vms/<id>/disk.qcow2`) is on the dead host. No other node
   rewrites that row when a peer disappears.
3. **Why you cannot just migrate it.** A migration is driven by the source
   node (`/vms/{id}/migrate` on the source; the destination only accepts an
   `-incoming` connection). No source qemu, no migration.
4. **If the host comes back.** On start, the api-server reconciles its own rows
   against its own pidfiles: a row marked `running` with no live qemu becomes
   `stopped`, a live qemu is re-adopted into the manager (so stop/delete still
   reach it), and a migration destination whose registration goroutine died
   before it wrote a row is registered from its `config.json`. That is
   reconcile-on-restart (commit `bebc0071`), not HA: it repairs bookkeeping,
   it never restarts a guest somewhere else.
5. **If the host does not come back.** The guest is gone unless you snapshotted
   its disks. Operator actions: restore the control-plane database onto the
   surviving cluster from a `backup-db.sh` dump if the database itself was lost;
   restore the disk image from your host-level snapshot onto a live node, create
   the VM there with the same disk; clear the stale row from any node with
   `DELETE /api/vms/{id}` (the delete path only stops/deletes a VM the *local*
   manager knows, so it cannot reach the dead host's disks).
6. **What we do not have:** no fencing/quorum, no replicated guest storage, no
   automatic restart of guests on a peer, no cross-node disk checksums. Treat a
   node with running guests as a single point of failure and size the blast
   radius accordingly.

## 8. Soak test (canary)

`scripts/fabric/two-node-fabric-test.sh` is the end-to-end proof: two real
api-servers (one inside a network namespace behind a `tc`-shaped veth), real
postgres, real qemu, a signed join, then membership, peer-job, admission-queue,
cross-node migration and usage-metering assertions against live traffic. One
green run proves a commit works once; the canary reruns it, because the fabric's
real failure modes (heartbeat races, admission-queue state, migration reconcile
across a restart) are the kind that pass by luck.

```bash
sudo deploy/scripts/soak-test.sh              # default: 2 consecutive clean runs, 14+ assertions
sudo deploy/scripts/soak-test.sh --iterations 5 --consecutive 3 --min-pass 14
```

Each iteration runs the harness with `FABRIC_REQUIRE_ALL=1` (a skipped
assertion counts as a failure) and `FABRIC_KEEP=1` (nodes, netns, veth pair and
databases stay up for inspection), shifting `FABRIC_PORT_A/B` per iteration so
preserved nodes do not collide. Per-iteration logs land in `/var/log/novacron/soak/`; the leftovers each run
keeps are printed at the end together with the exact commands that remove them
(`ip netns del`, `ip link del veth-fh-<pid>`, `DROP DATABASE`). `--clean` runs
with `FABRIC_KEEP=0` if you would rather leave nothing behind.

**Exit criteria for calling a build soak-clean:**

- every run reports `fail=0` **and** `skip=0` with `--min-pass 14`
  (14 = the assertion count once the migration-shape assertions land; bump
  `--min-pass` as new assertions are added — the number is checked, never
  assumed), and
- **two consecutive clean runs** (`--consecutive 2`), i.e. the canary exits `0`
  only after the fabric has been built, joined, migrated across and torn down
  twice in a row with nothing skipped.

A run that fails the criteria exits `1` and stops the loop immediately, leaving
the failing fabric up — inspect the preserved nodes and the log before cleaning
up. The script exits `2` without starting anything when sudo, postgres, or the
harness itself is missing.

## 9. Hardening: what the units do, and what they deliberately do not

Both units set `NoNewPrivileges`, `ProtectSystem=strict` with
`ReadWritePaths=/var/lib/novacron /var/log/novacron`, `ProtectHome`,
`PrivateTmp`, `ProtectKernelTunables`, `ProtectKernelModules`,
`ProtectControlGroups`, `RestrictRealtime`, `RestrictSUIDSGID`,
`LockPersonality`, `CapabilityBoundingSet=CAP_NET_BIND_SERVICE`,
`LimitNOFILE=65536`, `TasksMax=16384`, `Restart=on-failure`,
`TimeoutStopSec=45`, and journald logging.

Deliberate omissions, each because a directive that breaks the guests is worse
than no directive (the previous `systemd/novacron-api.service` had three of
these wrong):

- **No `MemoryMax`/`CPUQuota`.** `qemu-system-*` runs in this unit's cgroup, so
  any limit sized for the API process throttles or OOM-kills the guests. Cap
  guests through per-VM configuration instead.
- **No `PrivateDevices` and no `MemoryDenyWriteExecute`.** The first hides
  `/dev/kvm` from those children; the second breaks qemu's TCG (no-KVM) JIT.
- **No `AppArmorProfile`.** systemd fails the start when the profile is not
  already loaded in the kernel, so the directive is only honest once the
  operator has run `apparmor_parser -r apparmor/novacron-api` (that profile
  confines `/opt/novacron/bin/api-server`; it is not confined by default, and
  nothing in this repo installs it).
- **No `ExecReload`.** The server handles `SIGTERM`/`SIGINT` only — a `SIGHUP`
  reload would kill it. Reload means `systemctl restart`.
- **No capability beyond `CAP_NET_BIND_SERVICE`,** which is only there so you
  can move the API below 1024. `CAP_NET_ADMIN`/`CAP_SYS_ADMIN` are *not* needed
  by this binary: guests use qemu's SLIRP user networking (no tap devices), and
  the netns/veth/tc work belongs to the harness and to you, not to the service.
  The removed `CAP_DAC_READ_SEARCH` is now a trade-off you can see: the
  filesystem/secrets scan endpoint reads operator-supplied paths as the
  `novacron` user and therefore cannot read files that user cannot read.

To tighten further per host (e.g. a compute node with no admin API exposure),
use a drop-in:

```bash
sudo systemctl edit novacron-fabric-peer
# [Service]
# Environment=API_PORT=9000
# NOVACRON_NODE_ID=...          # if two roles share one host
# ReadWritePaths=/srv/novacron  # if STORAGE_PATH moved
```

## 10. Permissions

`install-node.sh`, `join-cluster.sh` (in its default mode), `backup-db.sh`,
`restore-db.sh` and `soak-test.sh` all need **root** — they write
`/etc/novacron` (0600 secrets), `/opt/novacron`, `/var/lib/novacron`, systemd
units, and the harness additionally needs passwordless sudo for netns/veth/tc.
`join-cluster.sh --env-file <file> --no-restart` and `--dry-run`, plus all
`--help` paths, run unprivileged. `systemd-analyze verify` and `bash -n` need no
privileges.

## 11. Files

| File | Purpose |
|---|---|
| `systemd/novacron-api-server.service` | control-plane unit (8090) |
| `systemd/novacron-fabric-peer.service` | compute-node unit (9000) |
| `scripts/install-node.sh` | install a role: prereqs, secrets, user, dirs, binaries, migrations, units, health |
| `scripts/join-cluster.sh` | signed join to a seed; verifies it took effect; persists `NOVACRON_JOIN_PEERS` |
| `scripts/backup-db.sh` | pg_dump custom-format backup + checksum + schema version |
| `scripts/restore-db.sh` | verified restore with a service-active and non-empty-DB guard |
| `scripts/soak-test.sh` | canary loop around the two-node harness with the exit criteria |