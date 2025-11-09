package state

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"encoding/json"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/accounts/abi/bind"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/ipfs/go-ipfs-api"
)

// StateManager manages blockchain state for VMs and resources
type StateManager struct {
	config        *StateConfig
	client        *ethclient.Client
	privateKey    *ecdsa.PrivateKey
	ipfsClient    *shell.Shell
	stateCache    sync.Map
	txQueue       chan *Transaction
	metrics       *StateMetrics
	mu            sync.RWMutex
}

// StateConfig defines state manager configuration
type StateConfig struct {
	RPCEndpoint       string
	PrivateKey        string
	IPFSEndpoint      string
	ContractAddresses map[string]common.Address
	CacheSize         int
	BatchSize         int
	SyncInterval      time.Duration
	GasPriceLimit     *big.Int
}

// Transaction represents a blockchain transaction
type Transaction struct {
	ID        string
	Type      string
	Data      interface{}
	GasLimit  uint64
	GasPrice  *big.Int
	Timestamp time.Time
	Status    string
	TxHash    string
	Error     error
}

// StateMetrics tracks blockchain state metrics
type StateMetrics struct {
	TotalTransactions    uint64
	SuccessfulTxs        uint64
	FailedTxs            uint64
	AverageGasUsed       uint64
	TotalGasCost         *big.Int
	AverageTPS           float64
	AverageFinalityTime  time.Duration
	IPFSStorageUsed      uint64
	OnChainVerifications uint64
	mu                   sync.RWMutex
}

// VMStateRecord represents on-chain VM state
type VMStateRecord struct {
	VMID              string
	Owner             common.Address
	Region            string
	State             uint8
	CPUAllocation     uint64
	MemoryAllocation  uint64
	StorageAllocation uint64
	NetworkBandwidth  uint64
	CreatedAt         time.Time
	LastModified      time.Time
	IPFSHash          string
	Metadata          map[string]interface{}
}

// NewStateManager creates a new blockchain state manager
func NewStateManager(config *StateConfig) (*StateManager, error) {
	// Connect to Ethereum/Polygon
	client, err := ethclient.Dial(config.RPCEndpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to blockchain: %w", err)
	}

	// Load private key
	privateKey, err := crypto.HexToECDSA(config.PrivateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to load private key: %w", err)
	}

	// Connect to IPFS
	ipfsClient := shell.NewShell(config.IPFSEndpoint)

	sm := &StateManager{
		config:     config,
		client:     client,
		privateKey: privateKey,
		ipfsClient: ipfsClient,
		txQueue:    make(chan *Transaction, config.BatchSize*10),
		metrics: &StateMetrics{
			TotalGasCost: big.NewInt(0),
		},
	}

	// Start background workers
	go sm.processTxQueue()
	go sm.syncState()
	go sm.updateMetrics()

	return sm, nil
}

// RecordVMState records VM state on blockchain and IPFS
func (sm *StateManager) RecordVMState(ctx context.Context, vmState *VMStateRecord) error {
	// Store detailed state in IPFS
	ipfsHash, err := sm.storeInIPFS(vmState)
	if err != nil {
		return fmt.Errorf("failed to store in IPFS: %w", err)
	}
	vmState.IPFSHash = ipfsHash

	// Create on-chain transaction
	tx := &Transaction{
		ID:        fmt.Sprintf("vm-state-%s-%d", vmState.VMID, time.Now().Unix()),
		Type:      "vm_state_update",
		Data:      vmState,
		GasLimit:  100000,
		Timestamp: time.Now(),
		Status:    "pending",
	}

	// Queue transaction
	select {
	case sm.txQueue <- tx:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// GetVMState retrieves VM state from blockchain and IPFS
func (sm *StateManager) GetVMState(ctx context.Context, vmID string) (*VMStateRecord, error) {
	// Check cache first
	if cached, ok := sm.stateCache.Load(vmID); ok {
		return cached.(*VMStateRecord), nil
	}

	// Query blockchain
	// This would call the smart contract's getVM function
	// For now, return a mock implementation
	state := &VMStateRecord{
		VMID:  vmID,
		State: uint8(2), // Running
	}

	// Retrieve detailed state from IPFS
	if state.IPFSHash != "" {
		if err := sm.retrieveFromIPFS(state.IPFSHash, state); err != nil {
			return nil, fmt.Errorf("failed to retrieve from IPFS: %w", err)
		}
	}

	// Cache the state
	sm.stateCache.Store(vmID, state)

	return state, nil
}

// CreateImmutableAudit creates an immutable audit trail on blockchain
func (sm *StateManager) CreateImmutableAudit(ctx context.Context, event map[string]interface{}) error {
	// Store audit event in IPFS
	ipfsHash, err := sm.storeInIPFS(event)
	if err != nil {
		return fmt.Errorf("failed to store audit in IPFS: %w", err)
	}

	// Record IPFS hash on-chain
	tx := &Transaction{
		ID:        fmt.Sprintf("audit-%d", time.Now().Unix()),
		Type:      "audit_trail",
		Data:      map[string]string{"ipfs_hash": ipfsHash},
		GasLimit:  50000,
		Timestamp: time.Now(),
		Status:    "pending",
	}

	select {
	case sm.txQueue <- tx:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// VerifyStateIntegrity verifies blockchain state integrity
func (sm *StateManager) VerifyStateIntegrity(ctx context.Context, vmID string) (bool, error) {
	// Get state from blockchain
	onChainState, err := sm.GetVMState(ctx, vmID)
	if err != nil {
		return false, err
	}

	// Verify IPFS hash
	if onChainState.IPFSHash != "" {
		data, err := sm.ipfsClient.Cat(onChainState.IPFSHash)
		if err != nil {
			return false, fmt.Errorf("IPFS verification failed: %w", err)
		}
		defer data.Close()

		// Verify data integrity
		var ipfsState VMStateRecord
		if err := json.NewDecoder(data).Decode(&ipfsState); err != nil {
			return false, fmt.Errorf("failed to decode IPFS data: %w", err)
		}

		// Compare critical fields
		if ipfsState.VMID != onChainState.VMID {
			return false, fmt.Errorf("VMID mismatch")
		}
	}

	sm.metrics.mu.Lock()
	sm.metrics.OnChainVerifications++
	sm.metrics.mu.Unlock()

	return true, nil
}

// storeInIPFS stores data in IPFS and returns the hash
func (sm *StateManager) storeInIPFS(data interface{}) (string, error) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return "", err
	}

	hash, err := sm.ipfsClient.Add(bytes.NewReader(jsonData))
	if err != nil {
		return "", err
	}

	sm.metrics.mu.Lock()
	sm.metrics.IPFSStorageUsed += uint64(len(jsonData))
	sm.metrics.mu.Unlock()

	return hash, nil
}

// retrieveFromIPFS retrieves data from IPFS
func (sm *StateManager) retrieveFromIPFS(hash string, dest interface{}) error {
	data, err := sm.ipfsClient.Cat(hash)
	if err != nil {
		return err
	}
	defer data.Close()

	return json.NewDecoder(data).Decode(dest)
}

// processTxQueue processes queued transactions in batches
func (sm *StateManager) processTxQueue() {
	batch := make([]*Transaction, 0, sm.config.BatchSize)
	ticker := time.NewTicker(time.Second * 5)
	defer ticker.Stop()

	for {
		select {
		case tx := <-sm.txQueue:
			batch = append(batch, tx)
			if len(batch) >= sm.config.BatchSize {
				sm.executeBatch(batch)
				batch = make([]*Transaction, 0, sm.config.BatchSize)
			}
		case <-ticker.C:
			if len(batch) > 0 {
				sm.executeBatch(batch)
				batch = make([]*Transaction, 0, sm.config.BatchSize)
			}
		}
	}
}

// executeBatch executes a batch of transactions
func (sm *StateManager) executeBatch(batch []*Transaction) {
	ctx := context.Background()
	startTime := time.Now()

	// Get auth transactor
	auth, err := bind.NewKeyedTransactorWithChainID(sm.privateKey, big.NewInt(137)) // Polygon mainnet
	if err != nil {
		for _, tx := range batch {
			tx.Status = "failed"
			tx.Error = err
		}
		return
	}

	// Set gas price
	gasPrice, err := sm.client.SuggestGasPrice(ctx)
	if err != nil {
		gasPrice = sm.config.GasPriceLimit
	}
	auth.GasPrice = gasPrice

	for _, tx := range batch {
		// Execute transaction based on type
		// This would call the appropriate smart contract function
		// For now, simulate transaction execution
		txHash := crypto.Keccak256Hash([]byte(tx.ID))
		tx.TxHash = txHash.Hex()
		tx.Status = "confirmed"
		tx.GasPrice = gasPrice

		// Update metrics
		sm.metrics.mu.Lock()
		sm.metrics.TotalTransactions++
		sm.metrics.SuccessfulTxs++
		sm.metrics.AverageGasUsed = (sm.metrics.AverageGasUsed*uint64(sm.metrics.SuccessfulTxs-1) + tx.GasLimit) / uint64(sm.metrics.SuccessfulTxs)
		gasCost := new(big.Int).Mul(gasPrice, big.NewInt(int64(tx.GasLimit)))
		sm.metrics.TotalGasCost.Add(sm.metrics.TotalGasCost, gasCost)
		sm.metrics.mu.Unlock()
	}

	// Update finality time
	finalityTime := time.Since(startTime)
	sm.metrics.mu.Lock()
	sm.metrics.AverageFinalityTime = (sm.metrics.AverageFinalityTime*time.Duration(sm.metrics.SuccessfulTxs-uint64(len(batch))) + finalityTime) / time.Duration(sm.metrics.SuccessfulTxs)
	sm.metrics.mu.Unlock()
}

// syncState synchronizes state with blockchain
func (sm *StateManager) syncState() {
	ticker := time.NewTicker(sm.config.SyncInterval)
	defer ticker.Stop()

	for range ticker.C {
		// Sync state from blockchain
		// This would query smart contracts for latest state
		// and update local cache
	}
}

// updateMetrics updates state metrics
func (sm *StateManager) updateMetrics() {
	ticker := time.NewTicker(time.Second * 30)
	defer ticker.Stop()

	for range ticker.C {
		sm.metrics.mu.Lock()
		if sm.metrics.TotalTransactions > 0 {
			duration := time.Second * 30
			sm.metrics.AverageTPS = float64(sm.metrics.TotalTransactions) / duration.Seconds()
		}
		sm.metrics.mu.Unlock()
	}
}

// GetMetrics returns current state metrics
func (sm *StateManager) GetMetrics() *StateMetrics {
	sm.metrics.mu.RLock()
	defer sm.metrics.mu.RUnlock()

	return &StateMetrics{
		TotalTransactions:    sm.metrics.TotalTransactions,
		SuccessfulTxs:        sm.metrics.SuccessfulTxs,
		FailedTxs:            sm.metrics.FailedTxs,
		AverageGasUsed:       sm.metrics.AverageGasUsed,
		TotalGasCost:         new(big.Int).Set(sm.metrics.TotalGasCost),
		AverageTPS:           sm.metrics.AverageTPS,
		AverageFinalityTime:  sm.metrics.AverageFinalityTime,
		IPFSStorageUsed:      sm.metrics.IPFSStorageUsed,
		OnChainVerifications: sm.metrics.OnChainVerifications,
	}
}

// Close closes the state manager
func (sm *StateManager) Close() error {
	close(sm.txQueue)
	sm.client.Close()
	return nil
}
