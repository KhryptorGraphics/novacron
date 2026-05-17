# VM I/O Backend Contracts

This document defines the backend contracts required before the NovaCron CLI can safely expose production `copy` and `port-forward` commands. The current backend has WebSocket contracts for console, logs, metrics, alerts, and migration events, but no VM file-transfer or port-forward tunnel surface.

## VM File Copy

### Endpoint

- `GET /api/ws/vms/{vmId}/copy`
- Alias: `GET /ws/vms/{vmId}/copy`
- Required role: `operator`
- Required query parameters:
  - `direction`: `upload` or `download`
  - `path`: absolute guest path
- Optional query parameters:
  - `mode`: octal file mode for uploads
  - `overwrite`: `true` or `false`, default `false`

### Frames

All frames are binary-safe WebSocket messages with a one-byte frame type prefix.

- `0x01` metadata JSON: `{"path":"/tmp/file","size":123,"mode":"0644","sha256":"..."}`
- `0x02` data chunk bytes
- `0x03` end-of-file JSON: `{"sha256":"...","bytes":123}`
- `0x04` error JSON: `{"code":"permission_denied","message":"..."}`
- `0x05` ack JSON: `{"bytes":123}`

### Acceptance Criteria

- Upload writes to a temporary file and atomically renames after checksum validation.
- Download fails if the path is a directory unless archive mode is explicitly added later.
- Server rejects relative paths, traversal, device files, and symlink escapes.
- Large transfers are backpressure-aware and bounded by per-tenant rate limits.
- Audit logs include user, VM ID, path, direction, byte count, checksum, and result.

## VM Port Forward

### Endpoint

- `GET /api/ws/vms/{vmId}/port-forward`
- Alias: `GET /ws/vms/{vmId}/port-forward`
- Required role: `operator`
- Required query parameters:
  - `port`: guest TCP port
- Optional query parameters:
  - `bind`: local bind address for audit/context only

### Frames

All frames are binary-safe WebSocket messages with a one-byte frame type prefix.

- `0x10` open JSON: `{"connectionId":"...","port":80}`
- `0x11` data: connection ID length byte, connection ID bytes, then payload bytes
- `0x12` close JSON: `{"connectionId":"...","reason":"eof"}`
- `0x13` error JSON: `{"connectionId":"...","code":"connect_failed","message":"..."}`
- `0x14` heartbeat JSON: `{"timestamp":"..."}`

### Acceptance Criteria

- Backend opens TCP connections from the VM network namespace, not from the API server namespace.
- Multiple concurrent local connections multiplex over one WebSocket.
- Idle connections time out with close frames and release guest-side sockets.
- Server enforces allowed port policy, tenant isolation, rate limits, and audit logging.
- CLI reports tunnel setup failure distinctly from per-connection guest failures.

## CLI Readiness Gate

`novacron copy` and `novacron port-forward` must remain disabled until these backend contracts are implemented and covered by integration tests. Tracking issue: `novacron-lmh`.
