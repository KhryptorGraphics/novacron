//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import "net/http"

// migrationAuthOK enforces the credential check that gates every
// /internal/migrate/* handler (the incoming-migration target endpoint and the
// abort endpoint in registerInternalMigrationRoutes). It FAILS CLOSED: with
// neither NOVACRON_MIGRATION_SECRET nor a NOVACRON_NODE_SECRETS entry for this
// node configured, no request is authorized. A node with no configured
// credential does not accept incoming migrations -- correct for a single-node
// deployment, which never receives them. Without this, any peer able to reach
// the api-server port could POST an attacker-controlled
// IncomingMigrationRequest and make the node launch an arbitrary qemu process
// (pre-auth RCE-class).
//
// The check itself lives in internalAuthOK (cluster_join.go), so every inbound
// node-to-node RPC shares one implementation: it accepts THIS node's own
// per-node credential when NOVACRON_NODE_SECRETS configures one, plus the
// fabric-wide NOVACRON_MIGRATION_SECRET while the fabric is mid-rollout, and
// compares in constant time via crypto/subtle so a mismatch can't be timed to
// recover the secret byte by byte. The caller must present the credential in
// the X-Migration-Secret header; with NOVACRON_NODE_SECRETS unset this is
// exactly the previous single-secret behaviour.
func migrationAuthOK(r *http.Request) bool {
	return internalAuthOK(r)
}
