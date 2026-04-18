# Phase 0 Runtime Truth Matrix

This document records the current truth-alignment boundary for NovaCron's default
`backend/core/cmd/novacron` runtime.

## Product vs Research Tracks

- Track A: product path for a federated P2P clustered hypervisor platform.
- Track B: research path for true cross-node CPU and memory pooling.

The default `cmd/novacron` runtime is part of Track A only. It is not a proof that
Track B exists or is production-ready.

## Landed In This Phase 0 Slice

- `cmd/novacron` now uses the real package constructors for:
  - `storage.StorageManager`
  - `network.NetworkManager`
  - `hypervisor.Hypervisor`
  - `vm.VMManager`
  - `scheduler.Scheduler`
- The config path is now typed and YAML-backed instead of returning an empty
  placeholder map.
- The command accepts YAML overrides for the fields needed by the current startup
  path:
  - `storage.base_path`
  - `hypervisor.id`
  - `hypervisor.name`
  - `hypervisor.role`
  - `hypervisor.data_dir`
  - `hypervisor.vm_config.type`
  - `hypervisor.vm_config.cpu_shares`
  - `hypervisor.vm_config.memory_mb`
  - `vm_manager.default_driver`
  - `vm_manager.default_vm_type`
  - `scheduler.minimum_node_count`
- The startup path now fails fast if the default KVM runtime still resolves to
  `vm.CoreStubDriver`.

## Explicit Non-Claims

- The default build does not provide a production KVM runtime yet.
- The default build does not provide true pooled CPU execution across nodes.
- The default build does not provide transparent pooled RAM across low-bandwidth
  peers.
- The default build does not provide a production distributed LUN backend yet.
- The migration manager in `cmd/novacron` is currently a local constructor wiring,
  not a complete WAN-grade migration subsystem.

## Current Capability Classification

### Product-core

- Typed config loading for the command bootstrap
- Real constructor-based startup wiring
- Explicit refusal to boot as a hypervisor on the stub KVM runtime

### Experimental

- Hypervisor package runtime and VM package internals outside the default command
  bootstrap
- Migration orchestration primitives
- Distributed storage plugins and alternative network subsystems

### Research

- Cross-node memory fabric
- VM state sharding for pooled execution
- Any feature described as pooled CPU or pooled RAM for a single guest

## Immediate Next Slices

1. Replace the default KVM `CoreStubDriver` path with a real production-backed KVM
   driver in the default build.
2. Wire authenticated node identity, membership, and low-bandwidth transport into
   the command bootstrap.
3. Promote one real distributed block/LUN backend into the default runtime.
4. Add checkpoint/cold-migration flows that are truthful under constrained links.
5. Run the Track B feasibility gate separately instead of implying it through the
   Track A runtime.
