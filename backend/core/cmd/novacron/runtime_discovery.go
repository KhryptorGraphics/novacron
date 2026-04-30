package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/federation"
	"github.com/shirou/gopsutil/v3/disk"
	gopsutilmem "github.com/shirou/gopsutil/v3/mem"
)

const runtimeDiscoveryPrivateKeyEnv = "NOVACRON_NODE_INVENTORY_PRIVATE_KEY"

type runtimeDiscoveryState struct {
	enabled      bool
	mode         string
	publicKey    string
	inventory    federation.SignedNodeInventory
	seedStatuses []runtimeDiscoverySeedStatus
	seedsByID    map[string]runtimeDiscoverySeed
}

type runtimeDiscoverySeedStatus struct {
	ID        string   `json:"id,omitempty"`
	Address   string   `json:"address"`
	PublicKey string   `json:"public_key,omitempty"`
	Tags      []string `json:"tags,omitempty"`
	State     string   `json:"state"`
	Reason    string   `json:"reason,omitempty"`
}

type runtimeDiscoveryInventoryResponse struct {
	Enabled   bool                           `json:"enabled"`
	Mode      string                         `json:"mode"`
	PublicKey string                         `json:"public_key"`
	Inventory federation.SignedNodeInventory `json:"inventory"`
	Seeds     []runtimeDiscoverySeedStatus   `json:"seeds,omitempty"`
}

type runtimeDiscoveryVerifyResponse struct {
	Valid     bool                       `json:"valid"`
	Seed      runtimeDiscoverySeedStatus `json:"seed"`
	Inventory federation.NodeInventory   `json:"inventory,omitempty"`
}

func newRuntimeDiscoveryState(config runtimeConfig, advertiseAddress string) (*runtimeDiscoveryState, error) {
	applyRuntimeServiceDefaults(&config)

	publicKey, privateKey, err := runtimeDiscoverySigningKey()
	if err != nil {
		return nil, fmt.Errorf("initialize node inventory signing key: %w", err)
	}

	inventory := federation.NodeInventory{
		Version:      federation.NodeInventoryVersionV1Alpha1,
		NodeID:       runtimeDiscoveryNodeID(config),
		ClusterID:    strings.TrimSpace(config.Auth.DefaultClusterID),
		Name:         strings.TrimSpace(config.Hypervisor.Name),
		Reachability: runtimeDiscoveryReachability(advertiseAddress),
		Capabilities: runtimeDiscoveryCapabilities(config),
		Resources:    runtimeDiscoveryResources(config),
		Storage:      runtimeDiscoveryStorage(config),
		VersionFlags: runtimeDiscoveryVersionFlags(config),
		IssuedAtUnix: time.Now().UTC().Unix(),
	}
	signedInventory, err := federation.SignNodeInventory(inventory, privateKey)
	if err != nil {
		return nil, fmt.Errorf("sign node inventory: %w", err)
	}

	state := &runtimeDiscoveryState{
		enabled:   runtimeDiscoveryEnabled(config),
		mode:      runtimeDiscoveryMode(config),
		publicKey: base64.StdEncoding.EncodeToString(publicKey),
		inventory: signedInventory,
		seedsByID: make(map[string]runtimeDiscoverySeed, len(config.Services.DiscoverySeeds)),
	}
	state.seedStatuses = runtimeDiscoverySeedStatuses(config.Services.DiscoverySeeds, state.seedsByID)
	return state, nil
}

func runtimeDiscoverySigningKey() (ed25519.PublicKey, ed25519.PrivateKey, error) {
	encodedPrivateKey := strings.TrimSpace(os.Getenv(runtimeDiscoveryPrivateKeyEnv))
	if encodedPrivateKey == "" {
		return ed25519.GenerateKey(rand.Reader)
	}
	rawPrivateKey, err := base64.StdEncoding.DecodeString(encodedPrivateKey)
	if err != nil {
		return nil, nil, fmt.Errorf("decode %s: %w", runtimeDiscoveryPrivateKeyEnv, err)
	}
	if len(rawPrivateKey) != ed25519.PrivateKeySize {
		return nil, nil, fmt.Errorf("%s must be base64 encoded ed25519 private key with %d raw bytes", runtimeDiscoveryPrivateKeyEnv, ed25519.PrivateKeySize)
	}
	privateKey := ed25519.PrivateKey(rawPrivateKey)
	publicKey, ok := privateKey.Public().(ed25519.PublicKey)
	if !ok || len(publicKey) != ed25519.PublicKeySize {
		return nil, nil, fmt.Errorf("%s did not contain a valid ed25519 private key", runtimeDiscoveryPrivateKeyEnv)
	}
	return publicKey, privateKey, nil
}

func runtimeDiscoveryEnabled(config runtimeConfig) bool {
	return runtimeServiceEnabled(config, "discovery") ||
		len(config.Services.DiscoverySeeds) > 0 ||
		runtimeDiscoveryMode(config) != "disabled"
}

func runtimeDiscoveryMode(config runtimeConfig) string {
	mode := strings.ToLower(strings.TrimSpace(config.Services.DiscoveryMode))
	if mode == "" {
		return "disabled"
	}
	return mode
}

func runtimeFederationEnabled(config runtimeConfig) bool {
	mode := strings.ToLower(strings.TrimSpace(config.Services.FederationMode))
	return runtimeServiceEnabled(config, "federation") || (mode != "" && mode != "disabled")
}

func runtimeDiscoveryNodeID(config runtimeConfig) string {
	if nodeID := strings.TrimSpace(config.Hypervisor.ID); nodeID != "" {
		return nodeID
	}
	return "local"
}

func runtimeDiscoveryReachability(advertiseAddress string) federation.NodeReachability {
	advertiseAddress = strings.TrimSpace(advertiseAddress)
	if advertiseAddress == "" {
		advertiseAddress = "127.0.0.1:0"
	}
	return federation.NodeReachability{
		AdvertiseAddress: advertiseAddress,
		APIAddress:       advertiseAddress,
	}
}

func runtimeDiscoveryCapabilities(config runtimeConfig) []string {
	capabilities := make([]string, 0, len(config.Services.EnabledServices)+2)
	for _, service := range config.Services.EnabledServices {
		service = strings.ToLower(strings.TrimSpace(service))
		if service == "" {
			continue
		}
		capabilities = append(capabilities, "service:"+service)
	}
	if runtimeDiscoveryEnabled(config) {
		capabilities = append(capabilities, "discovery:signed-inventory")
	}
	if runtimeFederationEnabled(config) {
		capabilities = append(capabilities, "federation:trusted-seed")
	}
	sort.Strings(capabilities)
	return capabilities
}

func runtimeDiscoveryResources(config runtimeConfig) federation.NodeResourceInventory {
	resources := federation.NodeResourceInventory{
		CPUCores: runtime.NumCPU(),
	}
	if memoryStats, err := gopsutilmem.VirtualMemory(); err == nil && memoryStats != nil {
		resources.MemoryBytes = int64(memoryStats.Total)
	}
	storagePath := strings.TrimSpace(config.Storage.BasePath)
	if storagePath != "" {
		if diskUsage, err := disk.Usage(storagePath); err == nil && diskUsage != nil {
			resources.StorageBytes = int64(diskUsage.Total)
		}
	}
	return resources
}

func runtimeDiscoveryStorage(config runtimeConfig) []federation.NodeStorageInventory {
	storagePath := strings.TrimSpace(config.Storage.BasePath)
	if storagePath == "" {
		return nil
	}
	diskUsage, err := disk.Usage(storagePath)
	if err != nil || diskUsage == nil {
		return nil
	}
	return []federation.NodeStorageInventory{
		{
			Class:          "default",
			CapacityBytes:  int64(diskUsage.Total),
			AvailableBytes: int64(diskUsage.Free),
		},
	}
}

func runtimeDiscoveryVersionFlags(config runtimeConfig) []string {
	flags := []string{
		"runtime:" + strings.TrimSpace(config.Services.Version),
		"profile:" + strings.TrimSpace(config.Services.DeploymentProfile),
		"discovery:" + runtimeDiscoveryMode(config),
		"federation:" + strings.TrimSpace(config.Services.FederationMode),
		"migration:" + strings.TrimSpace(config.Services.MigrationMode),
		"auth:" + strings.TrimSpace(config.Services.AuthMode),
	}
	return federation.NormalizeNodeInventory(federation.NodeInventory{VersionFlags: flags}).VersionFlags
}

func runtimeDiscoverySeedStatuses(seeds []runtimeDiscoverySeed, seedsByID map[string]runtimeDiscoverySeed) []runtimeDiscoverySeedStatus {
	if len(seeds) == 0 {
		return nil
	}
	statuses := make([]runtimeDiscoverySeedStatus, 0, len(seeds))
	for _, seed := range seeds {
		status := runtimeDiscoverySeedStatus{
			ID:        strings.TrimSpace(seed.ID),
			Address:   strings.TrimSpace(seed.Address),
			PublicKey: strings.TrimSpace(seed.PublicKey),
			Tags:      append([]string(nil), seed.Tags...),
		}
		switch {
		case status.Address == "":
			status.State = "invalid"
			status.Reason = "seed address is required"
		case status.PublicKey == "":
			status.State = "unverified"
			status.Reason = "seed public_key is required for signed inventory verification"
		case runtimeDecodeDiscoveryPublicKey(status.PublicKey) == nil:
			status.State = "invalid"
			status.Reason = "seed public_key must be base64 encoded ed25519 public key"
		default:
			status.State = "trusted"
		}
		if status.ID == "" {
			status.ID = status.Address
		}
		seedsByID[status.ID] = runtimeDiscoverySeed{
			ID:        status.ID,
			Address:   status.Address,
			PublicKey: status.PublicKey,
			Tags:      append([]string(nil), status.Tags...),
		}
		statuses = append(statuses, status)
	}
	return statuses
}

func runtimeDecodeDiscoveryPublicKey(encoded string) ed25519.PublicKey {
	raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(encoded))
	if err != nil || len(raw) != ed25519.PublicKeySize {
		return nil
	}
	return ed25519.PublicKey(raw)
}

func (s *runtimeDiscoveryState) verifyInventoryForSeed(seedID string, signed federation.SignedNodeInventory) (runtimeDiscoverySeedStatus, error) {
	if s == nil {
		return runtimeDiscoverySeedStatus{}, fmt.Errorf("discovery runtime is not initialized")
	}
	seedID = strings.TrimSpace(seedID)
	seed, ok := s.seedsByID[seedID]
	if !ok {
		return runtimeDiscoverySeedStatus{}, fmt.Errorf("unknown discovery seed %q", seedID)
	}
	status := runtimeDiscoverySeedStatus{
		ID:        seed.ID,
		Address:   seed.Address,
		PublicKey: seed.PublicKey,
		Tags:      append([]string(nil), seed.Tags...),
	}
	publicKey := runtimeDecodeDiscoveryPublicKey(seed.PublicKey)
	if publicKey == nil {
		status.State = "invalid"
		status.Reason = "seed public_key must be base64 encoded ed25519 public key"
		return status, fmt.Errorf("%s", status.Reason)
	}
	if err := federation.VerifySignedNodeInventory(signed, publicKey); err != nil {
		status.State = "rejected"
		status.Reason = err.Error()
		return status, err
	}
	status.State = "verified"
	return status, nil
}

func runtimeGetDiscoveryInventoryHandler(discovery *runtimeDiscoveryState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if discovery == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "discovery runtime is not initialized"})
			return
		}
		respondRuntimeJSON(w, http.StatusOK, runtimeDiscoveryInventoryResponse{
			Enabled:   discovery.enabled,
			Mode:      discovery.mode,
			PublicKey: discovery.publicKey,
			Inventory: discovery.inventory,
			Seeds:     discovery.seedStatuses,
		})
	}
}

func runtimeGetDiscoverySeedsHandler(discovery *runtimeDiscoveryState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if discovery == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "discovery runtime is not initialized"})
			return
		}
		respondRuntimeJSON(w, http.StatusOK, discovery.seedStatuses)
	}
}

func runtimeVerifyDiscoverySeedInventoryHandler(discovery *runtimeDiscoveryState) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if discovery == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "discovery runtime is not initialized"})
			return
		}
		var signed federation.SignedNodeInventory
		if err := json.NewDecoder(r.Body).Decode(&signed); err != nil {
			respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid signed node inventory payload"})
			return
		}
		status, err := discovery.verifyInventoryForSeed(mux.Vars(r)["id"], signed)
		if err != nil {
			respondRuntimeJSON(w, http.StatusUnauthorized, runtimeDiscoveryVerifyResponse{Valid: false, Seed: status})
			return
		}
		respondRuntimeJSON(w, http.StatusOK, runtimeDiscoveryVerifyResponse{
			Valid:     true,
			Seed:      status,
			Inventory: federation.NormalizeNodeInventory(signed.Inventory),
		})
	}
}
