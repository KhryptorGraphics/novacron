package federation

import (
	"crypto/ed25519"
	"crypto/rand"
	"testing"
)

func TestSignAndVerifyNodeInventory(t *testing.T) {
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}

	inventory := NodeInventory{
		NodeID: "node-a",
		Reachability: NodeReachability{
			AdvertiseAddress: "10.0.0.10:8090",
		},
		Capabilities: []string{"vm", "storage"},
		Resources: NodeResourceInventory{
			CPUCores:     8,
			MemoryBytes:  32 << 30,
			StorageBytes: 500 << 30,
		},
		IssuedAtUnix: 1710000000,
	}

	signed, err := SignNodeInventory(inventory, privateKey)
	if err != nil {
		t.Fatalf("SignNodeInventory returned error: %v", err)
	}
	if signed.Algorithm != NodeInventorySignatureEd25519 {
		t.Fatalf("signature algorithm = %q, want %q", signed.Algorithm, NodeInventorySignatureEd25519)
	}
	if err := VerifySignedNodeInventory(signed, publicKey); err != nil {
		t.Fatalf("VerifySignedNodeInventory returned error: %v", err)
	}

	signed.Inventory.Resources.CPUCores = 16
	if err := VerifySignedNodeInventory(signed, publicKey); err == nil {
		t.Fatal("expected tampered inventory signature verification to fail")
	}
}

func TestCanonicalNodeInventoryPayloadNormalizesLists(t *testing.T) {
	inventory := NodeInventory{
		NodeID: " node-a ",
		Reachability: NodeReachability{
			AdvertiseAddress: " 10.0.0.10:8090 ",
		},
		Capabilities: []string{"storage", "vm", "storage", ""},
		VersionFlags: []string{"wan", "trusted", "wan"},
		Storage: []NodeStorageInventory{
			{Class: "slow"},
			{Class: "fast"},
		},
	}

	normalized := NormalizeNodeInventory(inventory)
	if got, want := normalized.NodeID, "node-a"; got != want {
		t.Fatalf("node id = %q, want %q", got, want)
	}
	if got, want := normalized.Capabilities[0], "storage"; got != want {
		t.Fatalf("first capability = %q, want %q", got, want)
	}
	if got, want := normalized.Capabilities[1], "vm"; got != want {
		t.Fatalf("second capability = %q, want %q", got, want)
	}
	if got, want := normalized.Storage[0].Class, "fast"; got != want {
		t.Fatalf("first storage class = %q, want %q", got, want)
	}
}
