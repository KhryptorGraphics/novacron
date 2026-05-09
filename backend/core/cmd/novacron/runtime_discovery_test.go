package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/federation"
)

func TestRuntimeDiscoveryPublishesSignedLocalInventory(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.DiscoveryMode = "seeded"
	config.Services.EnabledServices = append(config.Services.EnabledServices, "discovery")
	expectedPublicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate discovery signing key: %v", err)
	}
	t.Setenv(runtimeDiscoveryPrivateKeyEnv, base64.StdEncoding.EncodeToString(privateKey))

	apiServer, err := initializeAPI(ctx, config, "127.0.0.1:0", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}
	defer apiServer.Shutdown(context.Background())

	response := getJSONResponse[runtimeDiscoveryInventoryResponse](t, apiServer, "/internal/runtime/v1/discovery/inventory")
	if !response.Enabled {
		t.Fatal("expected discovery inventory response to be enabled")
	}
	if got, want := response.Mode, "seeded"; got != want {
		t.Fatalf("discovery mode = %q, want %q", got, want)
	}
	if got, want := response.Inventory.Inventory.NodeID, "node-a"; got != want {
		t.Fatalf("inventory node id = %q, want %q", got, want)
	}
	if got, want := response.PublicKey, base64.StdEncoding.EncodeToString(expectedPublicKey); got != want {
		t.Fatalf("inventory public key = %q, want configured key %q", got, want)
	}
	publicKey := decodeTestDiscoveryPublicKey(t, response.PublicKey)
	if err := federation.VerifySignedNodeInventory(response.Inventory, publicKey); err != nil {
		t.Fatalf("published inventory did not verify with advertised public key: %v", err)
	}

	report := getJSONResponse[runtimeServiceReport](t, apiServer, "/internal/runtime/v1/services")
	serviceStates := make(map[string]runtimeServiceStatus, len(report.Services))
	for _, service := range report.Services {
		serviceStates[service.Name] = service
	}
	if got := serviceStates["discovery"].State; got != runtimeServiceStateRunning {
		t.Fatalf("discovery service state = %q, want %q", got, runtimeServiceStateRunning)
	}
	if got := serviceStates["federation"].State; got != runtimeServiceStateDisabled {
		t.Fatalf("federation service state = %q, want %q", got, runtimeServiceStateDisabled)
	}
}

func TestRuntimeDiscoveryVerifiesConfiguredSeedInventory(t *testing.T) {
	t.Parallel()

	seedPublicKey, seedPrivateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate seed key: %v", err)
	}
	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.DiscoveryMode = "seeded"
	config.Services.DiscoverySeeds = []runtimeDiscoverySeed{
		{
			ID:        "seed-a",
			Address:   "10.0.0.20:8090",
			PublicKey: base64.StdEncoding.EncodeToString(seedPublicKey),
			Tags:      []string{"trusted"},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	apiServer, err := initializeAPI(ctx, config, "127.0.0.1:0", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}
	defer apiServer.Shutdown(context.Background())

	signedInventory := signedSeedInventory(t, "seed-node", "10.0.0.20:8090", seedPrivateKey)
	verifyResponse := postDiscoveryJSONResponse[runtimeDiscoveryVerifyResponse](t, apiServer, "/internal/runtime/v1/discovery/seeds/seed-a/verify", signedInventory, http.StatusOK)
	if !verifyResponse.Valid {
		t.Fatal("expected seed inventory verification to be valid")
	}
	if got, want := verifyResponse.Seed.State, "verified"; got != want {
		t.Fatalf("seed state = %q, want %q", got, want)
	}
	if got, want := verifyResponse.Inventory.NodeID, "seed-node"; got != want {
		t.Fatalf("verified inventory node id = %q, want %q", got, want)
	}

	tampered := signedInventory
	tampered.Inventory.NodeID = "tampered-node"
	rejectedResponse := postDiscoveryJSONResponse[runtimeDiscoveryVerifyResponse](t, apiServer, "/internal/runtime/v1/discovery/seeds/seed-a/verify", tampered, http.StatusUnauthorized)
	if rejectedResponse.Valid {
		t.Fatal("expected tampered seed inventory verification to be rejected")
	}
	if got := rejectedResponse.Seed.State; got != "rejected" {
		t.Fatalf("tampered seed state = %q, want rejected", got)
	}
}

func TestRuntimeDiscoveryFetchesSeedInventoryAndRecordsMetrics(t *testing.T) {
	seedPublicKey, seedPrivateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate seed key: %v", err)
	}
	signedInventory := signedSeedInventory(t, "seed-node", "seed.example:8090", seedPrivateKey)
	seedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/runtime/v1/discovery/inventory" {
			http.NotFound(w, r)
			return
		}
		respondRuntimeJSON(w, http.StatusOK, runtimeDiscoveryInventoryResponse{
			Enabled:   true,
			Mode:      "seeded",
			PublicKey: base64.StdEncoding.EncodeToString(seedPublicKey),
			Inventory: signedInventory,
		})
	}))
	defer seedServer.Close()

	config := defaultRuntimeConfig("node-a", t.TempDir())
	config.Services.DiscoveryMode = "seeded"
	config.Services.DiscoverySeeds = []runtimeDiscoverySeed{
		{
			ID:        "seed-a",
			Address:   strings.TrimPrefix(seedServer.URL, "http://"),
			PublicKey: base64.StdEncoding.EncodeToString(seedPublicKey),
			Tags:      []string{"trusted"},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	apiServer, err := initializeAPI(ctx, config, "127.0.0.1:0", nil, nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("initializeAPI returned error: %v", err)
	}
	defer apiServer.Shutdown(context.Background())

	deadline := time.Now().Add(2 * time.Second)
	for {
		response := getJSONResponse[runtimeDiscoveryInventoryResponse](t, apiServer, "/internal/runtime/v1/discovery/inventory")
		if len(response.Seeds) == 1 && response.Seeds[0].State == "verified" {
			seedStatus := response.Seeds[0]
			if seedStatus.LastCheckUnix == 0 || seedStatus.LastVerifyUnix == 0 {
				t.Fatalf("expected seed check timestamps, got %#v", seedStatus)
			}
			if seedStatus.RTTMillis <= 0 {
				t.Fatalf("expected measured RTT, got %#v", seedStatus)
			}
			if seedStatus.ThroughputMbps <= 0 {
				t.Fatalf("expected measured throughput, got %#v", seedStatus)
			}
			network, ok := response.Inventory.Inventory.Network["seed-node"]
			if !ok {
				t.Fatalf("expected local signed inventory network metrics for seed-node, got %#v", response.Inventory.Inventory.Network)
			}
			if network.RTTMillis <= 0 || network.BandwidthMbps <= 0 || network.PacketLoss != 0 {
				t.Fatalf("unexpected network metrics: %#v", network)
			}
			publicKey := decodeTestDiscoveryPublicKey(t, response.PublicKey)
			if err := federation.VerifySignedNodeInventory(response.Inventory, publicKey); err != nil {
				t.Fatalf("updated local inventory did not verify after metrics injection: %v", err)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("seed inventory was not verified before deadline")
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func signedSeedInventory(t *testing.T, nodeID, address string, privateKey ed25519.PrivateKey) federation.SignedNodeInventory {
	t.Helper()
	signed, err := federation.SignNodeInventory(federation.NodeInventory{
		Version: federation.NodeInventoryVersionV1Alpha1,
		NodeID:  nodeID,
		Reachability: federation.NodeReachability{
			AdvertiseAddress: address,
			APIAddress:       address,
		},
		IssuedAtUnix: time.Now().UTC().Unix(),
	}, privateKey)
	if err != nil {
		t.Fatalf("sign seed inventory: %v", err)
	}
	return signed
}

func decodeTestDiscoveryPublicKey(t *testing.T, encoded string) ed25519.PublicKey {
	t.Helper()
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatalf("decode public key: %v", err)
	}
	if len(raw) != ed25519.PublicKeySize {
		t.Fatalf("public key length = %d, want %d", len(raw), ed25519.PublicKeySize)
	}
	return ed25519.PublicKey(raw)
}

func postDiscoveryJSONResponse[T any](t *testing.T, apiServer *APIServer, path string, payload interface{}, expectedStatus int) T {
	t.Helper()

	var zero T
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal POST payload: %v", err)
	}
	url := fmt.Sprintf("http://%s%s", apiServer.address, path)
	response, err := http.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s returned error: %v", url, err)
	}
	defer response.Body.Close()

	if response.StatusCode != expectedStatus {
		responseBody, _ := io.ReadAll(response.Body)
		t.Fatalf("POST %s status = %d, want %d: %s", url, response.StatusCode, expectedStatus, strings.TrimSpace(string(responseBody)))
	}
	if err := json.NewDecoder(response.Body).Decode(&zero); err != nil {
		t.Fatalf("decode %s response: %v", url, err)
	}
	return zero
}
