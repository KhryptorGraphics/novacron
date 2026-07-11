package main

import (
	"net/http/httptest"
	"os"
	"testing"
)

// TestInternalSecretOKFailsClosed proves the cluster internal-RPC guard
// (/internal/vms/create, /internal/cluster/capacity) rejects requests when no
// NOVACRON_MIGRATION_SECRET is configured. Discriminator: the prior
// `secret == "" || header == secret` body returned true (fail-open) with no
// secret set, which would let this test's first assertion pass a request through.
func TestInternalSecretOKFailsClosed(t *testing.T) {
	old, had := os.LookupEnv("NOVACRON_MIGRATION_SECRET")
	if had {
		defer os.Setenv("NOVACRON_MIGRATION_SECRET", old)
	} else {
		defer os.Unsetenv("NOVACRON_MIGRATION_SECRET")
	}

	os.Unsetenv("NOVACRON_MIGRATION_SECRET")
	r := httptest.NewRequest("POST", "/internal/vms/create", nil)
	if internalSecretOK(r) {
		t.Fatal("fail-open: internalSecretOK returned true with no secret configured; want false")
	}

	os.Setenv("NOVACRON_MIGRATION_SECRET", "s3cr3t-value")
	if internalSecretOK(r) {
		t.Fatal("internalSecretOK returned true with a missing header; want false")
	}
	r.Header.Set("X-Migration-Secret", "wrong")
	if internalSecretOK(r) {
		t.Fatal("internalSecretOK returned true with a wrong header; want false")
	}
	r.Header.Set("X-Migration-Secret", "s3cr3t-value")
	if !internalSecretOK(r) {
		t.Fatal("internalSecretOK returned false with the correct header; want true")
	}
}
