package federation

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

const (
	NodeInventoryVersionV1Alpha1  = "v1alpha1"
	NodeInventorySignatureEd25519 = "ed25519"
)

// NodeInventory is the canonical signed discovery payload shared by discovery,
// federation, placement, and UI surfaces.
type NodeInventory struct {
	Version      string                        `json:"version"`
	NodeID       string                        `json:"node_id"`
	ClusterID    string                        `json:"cluster_id,omitempty"`
	Name         string                        `json:"name,omitempty"`
	Reachability NodeReachability              `json:"reachability"`
	Capabilities []string                      `json:"capabilities,omitempty"`
	Resources    NodeResourceInventory         `json:"resources"`
	Storage      []NodeStorageInventory        `json:"storage,omitempty"`
	VersionFlags []string                      `json:"version_flags,omitempty"`
	Network      map[string]NodeNetworkMetrics `json:"network,omitempty"`
	IssuedAtUnix int64                         `json:"issued_at_unix"`
}

type NodeReachability struct {
	AdvertiseAddress  string `json:"advertise_address"`
	APIAddress        string `json:"api_address,omitempty"`
	FederationAddress string `json:"federation_address,omitempty"`
	NATType           string `json:"nat_type,omitempty"`
}

type NodeResourceInventory struct {
	CPUCores     int   `json:"cpu_cores"`
	MemoryBytes  int64 `json:"memory_bytes"`
	StorageBytes int64 `json:"storage_bytes"`
	VMs          int   `json:"vms"`
	NetworkPools int   `json:"network_pools"`
}

type NodeStorageInventory struct {
	Class          string `json:"class"`
	CapacityBytes  int64  `json:"capacity_bytes"`
	AvailableBytes int64  `json:"available_bytes"`
}

type NodeNetworkMetrics struct {
	RTTMillis      float64 `json:"rtt_millis"`
	BandwidthMbps  float64 `json:"bandwidth_mbps"`
	PacketLoss     float64 `json:"packet_loss"`
	MeasuredAtUnix int64   `json:"measured_at_unix"`
}

type SignedNodeInventory struct {
	Inventory NodeInventory `json:"inventory"`
	Algorithm string        `json:"algorithm"`
	Signature string        `json:"signature"`
}

func CanonicalNodeInventoryPayload(inventory NodeInventory) ([]byte, error) {
	normalized := NormalizeNodeInventory(inventory)
	if strings.TrimSpace(normalized.Version) == "" {
		normalized.Version = NodeInventoryVersionV1Alpha1
	}
	if strings.TrimSpace(normalized.NodeID) == "" {
		return nil, fmt.Errorf("node inventory node_id is required")
	}
	if strings.TrimSpace(normalized.Reachability.AdvertiseAddress) == "" {
		return nil, fmt.Errorf("node inventory reachability.advertise_address is required")
	}
	return json.Marshal(normalized)
}

func SignNodeInventory(inventory NodeInventory, privateKey ed25519.PrivateKey) (SignedNodeInventory, error) {
	if len(privateKey) != ed25519.PrivateKeySize {
		return SignedNodeInventory{}, fmt.Errorf("invalid ed25519 private key size")
	}
	normalized := NormalizeNodeInventory(inventory)
	if strings.TrimSpace(normalized.Version) == "" {
		normalized.Version = NodeInventoryVersionV1Alpha1
	}
	payload, err := CanonicalNodeInventoryPayload(inventory)
	if err != nil {
		return SignedNodeInventory{}, err
	}
	signature := ed25519.Sign(privateKey, payload)
	return SignedNodeInventory{
		Inventory: normalized,
		Algorithm: NodeInventorySignatureEd25519,
		Signature: base64.StdEncoding.EncodeToString(signature),
	}, nil
}

func VerifySignedNodeInventory(signed SignedNodeInventory, publicKey ed25519.PublicKey) error {
	if signed.Algorithm != NodeInventorySignatureEd25519 {
		return fmt.Errorf("unsupported node inventory signature algorithm: %s", signed.Algorithm)
	}
	if len(publicKey) != ed25519.PublicKeySize {
		return fmt.Errorf("invalid ed25519 public key size")
	}
	signature, err := base64.StdEncoding.DecodeString(signed.Signature)
	if err != nil {
		return fmt.Errorf("decode node inventory signature: %w", err)
	}
	payload, err := CanonicalNodeInventoryPayload(signed.Inventory)
	if err != nil {
		return err
	}
	if !ed25519.Verify(publicKey, payload, signature) {
		return fmt.Errorf("node inventory signature verification failed")
	}
	return nil
}

func NormalizeNodeInventory(inventory NodeInventory) NodeInventory {
	inventory.Version = strings.TrimSpace(inventory.Version)
	inventory.NodeID = strings.TrimSpace(inventory.NodeID)
	inventory.ClusterID = strings.TrimSpace(inventory.ClusterID)
	inventory.Name = strings.TrimSpace(inventory.Name)
	inventory.Reachability.AdvertiseAddress = strings.TrimSpace(inventory.Reachability.AdvertiseAddress)
	inventory.Reachability.APIAddress = strings.TrimSpace(inventory.Reachability.APIAddress)
	inventory.Reachability.FederationAddress = strings.TrimSpace(inventory.Reachability.FederationAddress)
	inventory.Reachability.NATType = strings.TrimSpace(inventory.Reachability.NATType)
	inventory.Capabilities = normalizeInventoryStringList(inventory.Capabilities)
	inventory.VersionFlags = normalizeInventoryStringList(inventory.VersionFlags)
	for i := range inventory.Storage {
		inventory.Storage[i].Class = strings.TrimSpace(inventory.Storage[i].Class)
	}
	sort.Slice(inventory.Storage, func(i, j int) bool {
		return inventory.Storage[i].Class < inventory.Storage[j].Class
	})
	return inventory
}

func normalizeInventoryStringList(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	normalized := make([]string, 0, len(values))
	for _, value := range values {
		item := strings.TrimSpace(value)
		if item == "" {
			continue
		}
		if _, exists := seen[item]; exists {
			continue
		}
		seen[item] = struct{}{}
		normalized = append(normalized, item)
	}
	sort.Strings(normalized)
	return normalized
}
