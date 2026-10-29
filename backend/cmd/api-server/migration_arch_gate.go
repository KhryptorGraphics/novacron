package main

// Migration architecture gate.
//
// Live migration only works between hosts of the SAME CPU architecture (an
// aarch64 guest state cannot resume on x86_64 and vice versa), so every entry
// point that admits a migration refuses a known mismatch up front with a clear
// 4xx instead of starting work that fails later or half-way:
//
//	POST /vms/{id}/migrate        sync route          (registerVMMigrateRoute)
//	POST /vms/{id}/migrate/async  async job route     (registerVMMigrateAsyncRoutes)
//	POST /transfers               transfer admission  (registerFabricTransferRoutes)
//	queueMigrationTransfer        node drain + VM healing
//
// Drain/healing target selection (migrationCandidatesFunc) additionally SKIPS
// incompatible peers, so a mixed-arch fabric still drains onto the peers that
// can take the VM. fabric jobs / work units create fresh Process-driver VMs on
// the placed node (no guest state moves), so they are deliberately not gated.
//
// Source arch is this node's: a migration can only start for a VM this node's
// manager owns (GetVM must succeed locally). Destination arch is the peer's own
// NodeCapacity.Arch, read live from the /internal/cluster/capacity RPC that join
// and the heartbeat already use -- no schema change. A destination whose arch is
// unknown (unreachable, unregistered, or an older build that does not report
// one) is NOT blocked: it is logged and surfaced as a "warnings" entry on the
// response, because the migration itself needs that same peer to answer.
//
// ponytail: admission-time check only, no cache; a transient capacity-probe
// failure degrades to "unknown" (allowed + warned). Upgrade = remember the last
// successful arch per peer from the heartbeat if that window ever matters.

import (
	"fmt"
	"log"
	"net/http"
	"runtime"
	"strings"

	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// archMismatchCode is the machine-readable "code" of the refusal body.
const archMismatchCode = "migration_arch_mismatch"

// normalizeArch maps the spellings GOARCH, uname and distro tooling use onto one
// canonical name per architecture; "" means unknown/not reported.
func normalizeArch(raw string) string {
	switch a := strings.ToLower(strings.TrimSpace(raw)); a {
	case "amd64", "x86_64", "x86-64", "x64":
		return "x86_64"
	case "arm64", "aarch64", "armv8", "armv8a", "armv8-a":
		return "aarch64"
	case "386", "i386", "i486", "i586", "i686", "x86":
		return "i386"
	case "arm", "armv6l", "armv7", "armv7l", "armhf":
		return "arm"
	default:
		return a
	}
}

// archsIncompatible is the single comparison every gate decision goes through:
// true only for a KNOWN mismatch. An unknown ("" after normalizing) side is
// never a mismatch.
func archsIncompatible(a, b string) bool {
	na, nb := normalizeArch(a), normalizeArch(b)
	return na != "" && nb != "" && na != nb
}

// archMismatchError is the refusal for a cross-architecture migration.
type archMismatchError struct {
	SourceNode string `json:"source_node"`
	SourceArch string `json:"source_arch"`
	TargetNode string `json:"target_node"`
	TargetArch string `json:"target_arch"`
}

func (e *archMismatchError) Error() string {
	return fmt.Sprintf("cannot migrate across CPU architectures: source node %q is %s, target node %q is %s (live migration is same-architecture only)",
		e.SourceNode, e.SourceArch, e.TargetNode, e.TargetArch)
}

// write sends the refusal: HTTP 422 with {"error","code","source_node",
// "source_arch","target_node","target_arch"}.
func (e *archMismatchError) write(w http.ResponseWriter) {
	writeJSON(w, http.StatusUnprocessableEntity, map[string]interface{}{
		"error":       e.Error(),
		"code":        archMismatchCode,
		"source_node": e.SourceNode,
		"source_arch": e.SourceArch,
		"target_node": e.TargetNode,
		"target_arch": e.TargetArch,
	})
}

// migrationPreflight is the admission check handed to routes constructed with
// an injected executor (they have no VM manager of their own). nil disables the
// gate -- only tests do that; production wiring always passes
// newMigrationPreflight(vmManager).
type migrationPreflight func(targetNode, probeAddr string) (warning string, refusal *archMismatchError)

func (p migrationPreflight) check(targetNode, probeAddr string) (string, *archMismatchError) {
	if p == nil {
		return "", nil
	}
	return p(targetNode, probeAddr)
}

func newMigrationPreflight(vmManager *core_vm.VMManager) migrationPreflight {
	return func(targetNode, probeAddr string) (string, *archMismatchError) {
		return checkMigrationArch(vmManager, targetNode, probeAddr)
	}
}

// checkMigrationArch decides whether this node may migrate a VM to targetNode.
// probeAddr overrides the registered peer address when the caller already has
// the address the guest will actually be sent to (explicit target_addr).
func checkMigrationArch(vmManager *core_vm.VMManager, targetNode, probeAddr string) (string, *archMismatchError) {
	targetNode = strings.TrimSpace(targetNode)
	if targetNode == "" || targetNode == selfNodeID() {
		return "", nil
	}
	addr := strings.TrimSpace(probeAddr)
	if addr == "" && vmManager != nil {
		addr = vmManager.MigrationPeers()[targetNode]
	}
	var targetArch string
	if addr != "" {
		if capacity, err := fetchPeerCapacityFor(targetNode, addr); err == nil {
			targetArch = capacity.Arch
		} else {
			log.Printf("migration arch gate: capacity probe of %s (%s) failed: %v", targetNode, addr, err)
		}
	}
	return decideMigrationArch(targetNode, targetArch)
}

// decideMigrationArch is the decision half of checkMigrationArch, for callers
// that already hold the peer's reported arch (POST /transfers reads it from the
// nodeProfiles it fetches anyway, so it does not probe the peer twice). "" =
// the peer reported none.
func decideMigrationArch(targetNode, reportedArch string) (string, *archMismatchError) {
	targetNode = strings.TrimSpace(targetNode)
	if targetNode == "" || targetNode == selfNodeID() {
		return "", nil
	}
	targetArch := normalizeArch(reportedArch)
	sourceArch := normalizeArch(runtime.GOARCH)
	if archsIncompatible(sourceArch, targetArch) {
		return "", &archMismatchError{
			SourceNode: selfNodeID(), SourceArch: sourceArch,
			TargetNode: targetNode, TargetArch: targetArch,
		}
	}
	if targetArch == "" {
		warning := fmt.Sprintf("architecture of target node %q is unknown (unreachable, unregistered, or it did not report one); the same-architecture requirement was not verified", targetNode)
		log.Printf("migration arch gate: %s", warning)
		return warning, nil
	}
	return "", nil
}

// withMigrationWarning adds the optional "warnings" array to a success body.
func withMigrationWarning(body map[string]interface{}, warning string) map[string]interface{} {
	if warning != "" {
		body["warnings"] = []string{warning}
	}
	return body
}
