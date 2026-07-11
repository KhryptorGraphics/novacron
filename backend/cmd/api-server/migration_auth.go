//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"crypto/subtle"
	"net/http"
	"os"
)

// migrationAuthOK enforces the shared-secret check that gates every
// /internal/migrate/* handler (currently just the incoming-migration target
// endpoint in registerInternalMigrationRoutes). It FAILS CLOSED: with no
// NOVACRON_MIGRATION_SECRET configured, no request is authorized. A node
// with no configured migration secret does not accept incoming migrations --
// correct for a single-node deployment, which never receives them. Without
// this, any peer able to reach the api-server port could POST an
// attacker-controlled IncomingMigrationRequest and make the node launch an
// arbitrary qemu process (pre-auth RCE-class).
//
// When a secret IS configured, the caller must present a matching
// X-Migration-Secret header; the comparison runs in constant time via
// crypto/subtle.ConstantTimeCompare so a mismatch can't be timed to recover
// the secret byte by byte.
func migrationAuthOK(r *http.Request) bool {
	secret := os.Getenv("NOVACRON_MIGRATION_SECRET")
	if secret == "" {
		return false
	}
	provided := r.Header.Get("X-Migration-Secret")
	return subtle.ConstantTimeCompare([]byte(secret), []byte(provided)) == 1
}
