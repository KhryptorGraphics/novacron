# Canonical VM Spec

NovaCron's canonical VM contract is centered on `backend/core/vm.VMConfig`.

Required core fields:
- `name`
- `type`
- `cpu_shares`
- `memory_mb`
- `disk_size_gb`

Root disk fields:
- `image`
- `rootfs`
- `cloud_init_iso`

Tenant ownership fields:
- `owner_id`
- `tenant_id`

Network fields:
- `network_id`
- `network_attachments[]`

Storage attachment fields:
- `volume_attachments[]`

Placement and mobility policy fields:
- `placement`
- `migration`
- `replication`

Normalization rules:
- If `type` is omitted, NovaCron defaults the canonical contract to `kvm`.
- The legacy `tags.vm_type` field is still populated for compatibility, but `type` is authoritative.
- `name` is synchronized between the outer create request and the inner VM spec.
- `network_id` is derived from the primary network attachment when the explicit field is omitted.
- `owner_id` and `tenant_id` are projected into tags for compatibility with existing consumers.

Compatibility rules:
- Existing callers that still use `tags.vm_type` continue to work.
- Existing callers that omit `command` now work for KVM guests because the guest boots from its disk contract rather than a process command.
- Non-KVM callers must still provide a runnable command when their runtime requires it.
