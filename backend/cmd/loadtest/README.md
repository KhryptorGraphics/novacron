# loadtest

A real load-test client for the canonical `api-server`: concurrent VM
create/migrate/delete over the real HTTP API, with a pass/fail report
(error rate + latency percentiles per operation). It does not start
NovaCron itself -- start a real server first, then point this at it.

## Quick start: single node (create + delete only)

```bash
# 1. A real Postgres the api-server can create its schema in.
docker run -d --name novacron-lt-pg -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=novacron -p 127.0.0.1:25432:5432 postgres:13-alpine

# 2. A real api-server. Ports/paths below avoid clashing with anything else
#    already running on a dev box -- adjust freely.
#    NOTE: STORAGE_PATH must be a SHORT path. The KVM driver puts a UNIX
#    socket (monitor.sock/qmp.sock) under <STORAGE_PATH>/vms/<vm-id>/, and
#    UNIX socket paths have a hard 108-byte kernel limit -- a deeply nested
#    STORAGE_PATH (e.g. under a long session-scoped tmp dir) will make every
#    VM Start() fail with "UNIX socket path ... is too long". /tmp/nclt1 is
#    short enough; anything comparably short works.
cd backend/cmd/api-server
DB_URL="postgresql://postgres:postgres@127.0.0.1:25432/novacron?sslmode=disable" \
AUTH_SECRET="some-secret-at-least-16-chars-long" \
API_PORT=18090 WS_PORT=18091 STORAGE_PATH=/tmp/nclt1 \
go run . &

# 3. The load test. -db-url makes it seed a real "operator"-role account
#    directly in Postgres: self-registration via /auth/register only grants
#    role "user" (implies nothing in the RBAC hierarchy), which the VM
#    routes' require("operator", ...) rejects, so a load test that only
#    self-registers can never legitimately create/delete a VM.
cd backend/cmd/loadtest
go run . -api-url http://127.0.0.1:18090 \
  -db-url "postgresql://postgres:postgres@127.0.0.1:25432/novacron?sslmode=disable" \
  -concurrency 6 -creates 30
```

## Two nodes (adds the migrate phase)

Live migration needs a second api-server to migrate to. Give it its own
Postgres database (the cluster scheduler's best-fit create dispatch and
migrate both use a plain `owner_id` foreign key with no cross-node identity
sync, so the SAME operator user -- same numeric id -- must exist in both
databases; `-peer-db-url` takes care of that):

```bash
docker exec novacron-lt-pg psql -U postgres -c "CREATE DATABASE novacron2"

DB_URL="postgresql://postgres:postgres@127.0.0.1:25432/novacron2?sslmode=disable" \
AUTH_SECRET="some-secret-at-least-16-chars-long" \
API_PORT=18190 WS_PORT=18191 STORAGE_PATH=/tmp/nclt2 \
NOVACRON_MIGRATION_SECRET="shared-migrate-secret" \
go run . &   # node2, from backend/cmd/api-server

# node1 needs NOVACRON_PEERS + the same NOVACRON_MIGRATION_SECRET, so restart
# it (from step 2 above) with:
NOVACRON_PEERS="node2=127.0.0.1:18190" \
NOVACRON_MIGRATION_SECRET="shared-migrate-secret" \
... (same DB_URL/AUTH_SECRET/API_PORT/WS_PORT/STORAGE_PATH as before)

go run . -api-url http://127.0.0.1:18090 -peer-api-url http://127.0.0.1:18190 \
  -db-url "postgresql://postgres:postgres@127.0.0.1:25432/novacron?sslmode=disable" \
  -peer-db-url "postgresql://postgres:postgres@127.0.0.1:25432/novacron2?sslmode=disable" \
  -concurrency 6 -creates 30 -with-migrate -migrate-target node2 -migrate-count 5
```

`-peer-api-url` matters even without `-with-migrate`: NovaCron's cluster
scheduler does best-fit placement on EVERY create once a peer is registered
(`NOVACRON_PEERS`), not just on explicit migrate calls (see
`clusteredCreateHandler` in `cluster.go`) -- some of your "local" creates can
silently land on the peer. Delete/get are not cluster-dispatched the way
create is, so this tool tracks each VM's actual owning node (from the
create response's `placed_on` field, updated again on a successful migrate)
and deletes it through the right node's own API. Without `-peer-api-url`,
any VM the scheduler placed on the peer is reported as an unrecoverable
delete error rather than silently left running.

## Flags

Run `go run . -h` for the full list. The ones worth knowing about:

- `-concurrency`, `-creates`: worker count and total create operations.
- `-with-migrate`, `-migrate-target`, `-migrate-count`: opt-in migrate phase.
- `-max-error-rate`, `-max-p95-create-ms`, `-max-p95-delete-ms`,
  `-max-p95-migrate-ms`: pass/fail thresholds. Exit code is 1 if any phase
  breaches its threshold.
- `-image`, `-memory-mb`, `-disk-size-gb`: the VM spec used for every create.
  Defaults to a real cirros test image if it finds one at the same path the
  `backend/core/vm` real-qemu tests use (`~/novacron-run/images/` or
  `~/novacron-e2e/images/`); override with `-image` otherwise.

## What "real" means here

Every operation is real, not simulated: `create` runs `qemu-img convert` of
an actual base image into a fresh qcow2 disk on the server's real
filesystem; `migrate` is a genuine cross-process QEMU live migration (a
second `api-server` process receiving a real `-incoming` qemu over TCP);
`delete` tears down the real VM directory. There is no in-memory or mocked
driver involved.
