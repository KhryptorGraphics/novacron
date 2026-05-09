package main

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/federation"
	"github.com/shirou/gopsutil/v3/disk"
	gopsutilmem "github.com/shirou/gopsutil/v3/mem"
)

const runtimeDiscoveryPrivateKeyEnv = "NOVACRON_NODE_INVENTORY_PRIVATE_KEY"

type runtimeDiscoveryState struct {
	enabled   bool
	mode      string
	publicKey string

	mu           sync.RWMutex
	privateKey   ed25519.PrivateKey
	inventory    federation.SignedNodeInventory
	seedStatuses []runtimeDiscoverySeedStatus
	seedsByID    map[string]runtimeDiscoverySeed
	httpClient   *http.Client
	cancel       context.CancelFunc
	done         chan struct{}
}

type runtimeDiscoverySeedStatus struct {
	ID             string                         `json:"id,omitempty"`
	Address        string                         `json:"address"`
	PublicKey      string                         `json:"public_key,omitempty"`
	Tags           []string                       `json:"tags,omitempty"`
	State          string                         `json:"state"`
	Reason         string                         `json:"reason,omitempty"`
	LastCheckUnix  int64                          `json:"last_check_unix,omitempty"`
	LastVerifyUnix int64                          `json:"last_verified_unix,omitempty"`
	RTTMillis      float64                        `json:"rtt_millis,omitempty"`
	PacketLoss     float64                        `json:"packet_loss,omitempty"`
	ThroughputMbps float64                        `json:"throughput_mbps,omitempty"`
	Network        *federation.NodeNetworkMetrics `json:"network,omitempty"`
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
		enabled:    runtimeDiscoveryEnabled(config),
		mode:       runtimeDiscoveryMode(config),
		publicKey:  base64.StdEncoding.EncodeToString(publicKey),
		privateKey: privateKey,
		inventory:  signedInventory,
		seedsByID:  make(map[string]runtimeDiscoverySeed, len(config.Services.DiscoverySeeds)),
		httpClient: &http.Client{Timeout: 5 * time.Second},
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

func (s *runtimeDiscoveryState) Start(ctx context.Context) {
	if s == nil || !s.enabled || len(s.seedStatuses) == 0 {
		return
	}
	s.mu.Lock()
	if s.cancel != nil {
		s.mu.Unlock()
		return
	}
	runCtx, cancel := context.WithCancel(ctx)
	s.cancel = cancel
	s.done = make(chan struct{})
	s.mu.Unlock()

	go func() {
		defer close(s.done)
		s.fetchSeeds(runCtx)
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-runCtx.Done():
				return
			case <-ticker.C:
				s.fetchSeeds(runCtx)
			}
		}
	}()
}

func (s *runtimeDiscoveryState) Stop() {
	if s == nil {
		return
	}
	s.mu.Lock()
	cancel := s.cancel
	done := s.done
	s.cancel = nil
	s.done = nil
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if done != nil {
		<-done
	}
}

func (s *runtimeDiscoveryState) snapshotInventory() federation.SignedNodeInventory {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.inventory
}

func (s *runtimeDiscoveryState) snapshotSeedStatuses() []runtimeDiscoverySeedStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneRuntimeDiscoverySeedStatuses(s.seedStatuses)
}

func cloneRuntimeDiscoverySeedStatuses(statuses []runtimeDiscoverySeedStatus) []runtimeDiscoverySeedStatus {
	if len(statuses) == 0 {
		return nil
	}
	cloned := make([]runtimeDiscoverySeedStatus, len(statuses))
	for i, status := range statuses {
		cloned[i] = status
		cloned[i].Tags = append([]string(nil), status.Tags...)
		if status.Network != nil {
			network := *status.Network
			cloned[i].Network = &network
		}
	}
	return cloned
}

func (s *runtimeDiscoveryState) fetchSeeds(ctx context.Context) {
	statuses := s.snapshotSeedStatuses()
	for _, status := range statuses {
		if ctx.Err() != nil {
			return
		}
		if status.State == "invalid" || status.PublicKey == "" {
			continue
		}
		s.fetchSeed(ctx, status.ID)
	}
}

func (s *runtimeDiscoveryState) fetchSeed(ctx context.Context, seedID string) {
	seed, ok := s.seed(seedID)
	if !ok {
		return
	}
	started := time.Now()
	s.updateSeedStatus(seed.ID, func(status *runtimeDiscoverySeedStatus) {
		status.State = "fetching"
		status.Reason = ""
		status.LastCheckUnix = started.UTC().Unix()
	})

	inventoryURL, err := runtimeDiscoverySeedInventoryURL(seed.Address)
	if err != nil {
		s.markSeedFailure(seed.ID, "invalid", err.Error(), started)
		return
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, inventoryURL, nil)
	if err != nil {
		s.markSeedFailure(seed.ID, "invalid", err.Error(), started)
		return
	}
	client := s.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(request)
	if err != nil {
		if ctx.Err() != nil {
			return
		}
		s.markSeedFailure(seed.ID, "unreachable", err.Error(), started)
		return
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	elapsed := time.Since(started)
	if err != nil {
		s.markSeedFailure(seed.ID, "unreachable", err.Error(), started)
		return
	}
	if response.StatusCode != http.StatusOK {
		s.markSeedFailure(seed.ID, "unreachable", fmt.Sprintf("inventory fetch returned HTTP %d", response.StatusCode), started)
		return
	}
	var payload runtimeDiscoveryInventoryResponse
	if err := json.Unmarshal(body, &payload); err != nil {
		s.markSeedFailure(seed.ID, "invalid", "invalid discovery inventory response", started)
		return
	}
	status, err := s.verifyInventoryForSeed(seed.ID, payload.Inventory)
	if err != nil {
		return
	}
	throughputMbps := runtimeDiscoveryThroughputMbps(len(body), elapsed)
	network := federation.NodeNetworkMetrics{
		RTTMillis:      float64(elapsed.Microseconds()) / 1000,
		BandwidthMbps:  throughputMbps,
		PacketLoss:     0,
		MeasuredAtUnix: time.Now().UTC().Unix(),
	}
	status.LastCheckUnix = started.UTC().Unix()
	status.LastVerifyUnix = time.Now().UTC().Unix()
	status.RTTMillis = network.RTTMillis
	status.PacketLoss = network.PacketLoss
	status.ThroughputMbps = throughputMbps
	status.Network = &network
	s.storeVerifiedSeedStatus(status, payload.Inventory.Inventory.NodeID, network)
}

func runtimeDiscoverySeedInventoryURL(address string) (string, error) {
	address = strings.TrimSpace(address)
	if address == "" {
		return "", fmt.Errorf("seed address is required")
	}
	if !strings.Contains(address, "://") {
		address = "http://" + address
	}
	parsed, err := url.Parse(address)
	if err != nil {
		return "", err
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("seed address must include a host")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/internal/runtime/v1/discovery/inventory"
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

func runtimeDiscoveryThroughputMbps(bytes int, elapsed time.Duration) float64 {
	if bytes <= 0 || elapsed <= 0 {
		return 0
	}
	return (float64(bytes) * 8) / elapsed.Seconds() / 1_000_000
}

func (s *runtimeDiscoveryState) seed(seedID string) (runtimeDiscoverySeed, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	seed, ok := s.seedsByID[strings.TrimSpace(seedID)]
	return seed, ok
}

func (s *runtimeDiscoveryState) markSeedFailure(seedID, state, reason string, checkedAt time.Time) {
	packetLoss := 1.0
	s.updateSeedStatus(seedID, func(status *runtimeDiscoverySeedStatus) {
		status.State = state
		status.Reason = reason
		status.LastCheckUnix = checkedAt.UTC().Unix()
		status.PacketLoss = packetLoss
		status.Network = &federation.NodeNetworkMetrics{
			PacketLoss:     packetLoss,
			MeasuredAtUnix: time.Now().UTC().Unix(),
		}
	})
}

func (s *runtimeDiscoveryState) updateSeedStatus(seedID string, update func(*runtimeDiscoverySeedStatus)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.seedStatuses {
		if s.seedStatuses[i].ID == seedID {
			update(&s.seedStatuses[i])
			return
		}
	}
}

func (s *runtimeDiscoveryState) storeVerifiedSeedStatus(status runtimeDiscoverySeedStatus, seedNodeID string, network federation.NodeNetworkMetrics) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.seedStatuses {
		if s.seedStatuses[i].ID == status.ID {
			s.seedStatuses[i] = status
			break
		}
	}
	seedNodeID = strings.TrimSpace(seedNodeID)
	if seedNodeID == "" {
		return
	}
	inventory := s.inventory.Inventory
	if inventory.Network == nil {
		inventory.Network = make(map[string]federation.NodeNetworkMetrics)
	}
	inventory.Network[seedNodeID] = network
	signed, err := federation.SignNodeInventory(inventory, s.privateKey)
	if err == nil {
		s.inventory = signed
	}
}

func (s *runtimeDiscoveryState) verifyInventoryForSeed(seedID string, signed federation.SignedNodeInventory) (runtimeDiscoverySeedStatus, error) {
	if s == nil {
		return runtimeDiscoverySeedStatus{}, fmt.Errorf("discovery runtime is not initialized")
	}
	seedID = strings.TrimSpace(seedID)
	s.mu.RLock()
	seed, ok := s.seedsByID[seedID]
	s.mu.RUnlock()
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
	status.LastVerifyUnix = time.Now().UTC().Unix()
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
			Inventory: discovery.snapshotInventory(),
			Seeds:     discovery.snapshotSeedStatuses(),
		})
	}
}

func runtimeGetDiscoverySeedsHandler(discovery *runtimeDiscoveryState) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if discovery == nil {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "discovery runtime is not initialized"})
			return
		}
		respondRuntimeJSON(w, http.StatusOK, discovery.snapshotSeedStatuses())
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
