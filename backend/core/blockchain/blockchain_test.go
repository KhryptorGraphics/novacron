package blockchain

import (
	"context"
	"math/big"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"novacron/backend/core/blockchain/consensus"
	"novacron/backend/core/blockchain/contracts"
	"novacron/backend/core/blockchain/crosschain"
	"novacron/backend/core/blockchain/did"
	"novacron/backend/core/blockchain/governance"
	"novacron/backend/core/blockchain/l2"
	"novacron/backend/core/blockchain/metrics"
	"novacron/backend/core/blockchain/security"
	"novacron/backend/core/blockchain/state"
	"novacron/backend/core/blockchain/tokens"
)

func TestStateManager(t *testing.T) {
	config := &state.StateConfig{
		RPCEndpoint:  "http://localhost:8545",
		PrivateKey:   "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
		IPFSEndpoint: "http://localhost:5001",
		CacheSize:    1000,
		BatchSize:    100,
		SyncInterval: time.Minute,
		GasPriceLimit: big.NewInt(50000000000),
	}

	// Note: This test requires actual blockchain connection
	// In production, use a test network or mock
	t.Skip("Requires blockchain connection")

	sm, err := state.NewStateManager(config)
	require.NoError(t, err)
	defer sm.Close()

	// Test VM state recording
	vmState := &state.VMStateRecord{
		VMID:             "test-vm-1",
		Owner:            common.HexToAddress("0x1234567890123456789012345678901234567890"),
		Region:           "us-east-1",
		State:            2, // Running
		CPUAllocation:    4,
		MemoryAllocation: 8192,
	}

	err = sm.RecordVMState(context.Background(), vmState)
	assert.NoError(t, err)

	// Test state retrieval
	retrieved, err := sm.GetVMState(context.Background(), "test-vm-1")
	assert.NoError(t, err)
	assert.Equal(t, vmState.VMID, retrieved.VMID)
}

func TestContractOrchestrator(t *testing.T) {
	config := &contracts.OrchestratorConfig{
		RPCEndpoint:       "http://localhost:8545",
		PrivateKey:        "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
		ChainID:           big.NewInt(137), // Polygon
		GasLimit:          1000000,
		GasPrice:          big.NewInt(50000000000),
		MultiSigThreshold: 3,
		TimeLockDelay:     time.Hour * 24,
	}

	t.Skip("Requires blockchain connection")

	co, err := contracts.NewContractOrchestrator(config)
	require.NoError(t, err)
	defer co.Close()

	// Test VM creation
	vmOp := &contracts.VMLifecycleOperation{
		VMID:             "test-vm-1",
		Operation:        "create",
		Owner:            common.HexToAddress("0x1234567890123456789012345678901234567890"),
		Region:           "us-east-1",
		CPUAllocation:    4,
		MemoryAllocation: 8192,
	}

	txHash, err := co.CreateVM(context.Background(), vmOp)
	assert.NoError(t, err)
	assert.NotEmpty(t, txHash)
}

func TestDIDManager(t *testing.T) {
	config := &did.DIDConfig{
		Network:        "polygon",
		EnableZKProofs: true,
		CredentialTTL:  time.Hour * 24 * 365,
	}

	dm := did.NewDIDManager(config)

	// Test DID creation
	doc, privateKey, err := dm.CreateDID(context.Background(), "vm")
	require.NoError(t, err)
	assert.NotNil(t, doc)
	assert.NotNil(t, privateKey)
	assert.Contains(t, doc.ID, "did:novacron:polygon:")

	// Test DID resolution
	resolved, err := dm.ResolveDID(context.Background(), doc.ID)
	assert.NoError(t, err)
	assert.Equal(t, doc.ID, resolved.ID)

	// Test credential issuance
	claims := map[string]interface{}{
		"cpuAllocation":    4,
		"memoryAllocation": 8192,
		"region":           "us-east-1",
	}

	credential, err := dm.IssueCredential(context.Background(), doc.ID, doc.ID, claims, privateKey)
	assert.NoError(t, err)
	assert.NotNil(t, credential)
	assert.Equal(t, doc.ID, credential.Issuer)

	// Test credential verification
	valid, err := dm.VerifyCredential(context.Background(), credential)
	assert.NoError(t, err)
	assert.True(t, valid)

	// Test ZK proof creation
	zkProof, err := dm.CreateZKProof(context.Background(), "I have more than 4 CPUs", map[string]interface{}{"cpu": 8})
	assert.NoError(t, err)
	assert.NotNil(t, zkProof)

	// Test ZK proof verification
	valid, err = dm.VerifyZKProof(context.Background(), zkProof)
	assert.NoError(t, err)
	assert.True(t, valid)
}

func TestTokenManager(t *testing.T) {
	config := &tokens.TokenConfig{
		EnableTokens:       true,
		InitialSupply:      big.NewInt(1000000),
		StakingEnabled:     true,
		StakingRewardRate:  0.1, // 10% APR
		MarketplaceEnabled: true,
	}

	tm := tokens.NewTokenManager(config)

	user := common.HexToAddress("0x1234567890123456789012345678901234567890")

	// Test token minting
	err := tm.MintTokens(context.Background(), user, tokens.ResourceCPU, big.NewInt(1000))
	assert.NoError(t, err)

	balance := tm.GetBalance(user, tokens.ResourceCPU)
	assert.Equal(t, big.NewInt(1000), balance)

	// Test token transfer
	recipient := common.HexToAddress("0x0987654321098765432109876543210987654321")
	err = tm.TransferTokens(context.Background(), user, recipient, tokens.ResourceCPU, big.NewInt(100))
	assert.NoError(t, err)

	balance = tm.GetBalance(recipient, tokens.ResourceCPU)
	assert.Equal(t, big.NewInt(100), balance)

	// Test staking
	err = tm.StakeTokens(context.Background(), user, tokens.ResourceCPU, big.NewInt(500), time.Hour*24*30)
	assert.NoError(t, err)

	stakeInfo := tm.GetStakeInfo(user, tokens.ResourceCPU)
	assert.NotNil(t, stakeInfo)
	assert.Equal(t, big.NewInt(500), stakeInfo.Amount)

	// Test spot price
	price, err := tm.GetSpotPrice(tokens.ResourceCPU)
	assert.NoError(t, err)
	assert.NotNil(t, price)
}

func TestGovernanceManager(t *testing.T) {
	config := &governance.GovernanceConfig{
		ProposalThreshold: big.NewInt(100000),
		VotingPeriod:      time.Hour * 24 * 7,
		QuorumPercentage:  0.4,
		ExecutionDelay:    time.Hour * 24 * 2,
		EnableQuadratic:   true,
		EnableDelegation:  true,
	}

	gm := governance.NewGovernanceManager(config)

	proposer := common.HexToAddress("0x1234567890123456789012345678901234567890")

	// Mock token balance for proposer
	gm.SetTokenBalance(proposer, big.NewInt(150000))

	// Test proposal creation
	actions := []governance.ProposalAction{
		{
			Target:      common.HexToAddress("0x0987654321098765432109876543210987654321"),
			Value:       big.NewInt(0),
			Signature:   "updateConfig(uint256)",
			Calldata:    []byte{},
			Description: "Update system configuration",
		},
	}

	proposal, err := gm.CreateProposal(context.Background(), proposer, "Test Proposal", "This is a test proposal", actions)
	assert.NoError(t, err)
	assert.NotNil(t, proposal)
	assert.Equal(t, governance.ProposalStatePending, proposal.State)

	// Test voting
	voter := common.HexToAddress("0x1111111111111111111111111111111111111111")
	gm.SetTokenBalance(voter, big.NewInt(50000))

	// Fast-forward time to make proposal active
	proposal.State = governance.ProposalStateActive

	err = gm.CastVote(context.Background(), proposal.ID, voter, governance.VoteTypeFor, "I support this proposal")
	assert.NoError(t, err)

	// Test delegation
	delegator := common.HexToAddress("0x2222222222222222222222222222222222222222")
	err = gm.DelegateVote(context.Background(), delegator, voter)
	assert.NoError(t, err)
}

func TestCrossChainBridge(t *testing.T) {
	config := &crosschain.BridgeConfig{
		SupportedChains:    []string{"ethereum", "polygon", "solana"},
		ValidatorThreshold: 3,
		TransferTimeout:    time.Minute * 30,
	}

	ccb := crosschain.NewCrossChainBridge(config)

	sender := common.HexToAddress("0x1234567890123456789012345678901234567890")
	recipient := common.HexToAddress("0x0987654321098765432109876543210987654321")

	// Test bridge initiation
	transfer, err := ccb.InitiateBridge(
		context.Background(),
		"ethereum",
		"polygon",
		sender,
		recipient,
		big.NewInt(1000),
		common.HexToAddress("0x0000000000000000000000000000000000000000"),
	)

	assert.NoError(t, err)
	assert.NotNil(t, transfer)
	assert.Equal(t, crosschain.TransferStatusPending, transfer.Status)

	// Test transfer retrieval
	retrieved, err := ccb.GetTransfer(transfer.ID)
	assert.NoError(t, err)
	assert.Equal(t, transfer.ID, retrieved.ID)
}

func TestL2Manager(t *testing.T) {
	config := &l2.L2Config{
		EnableL2:     true,
		L2Type:       "optimistic",
		BatchSize:    1000,
		SequencerURL: "http://localhost:8080",
		TargetTPS:    10000,
	}

	l2m := l2.NewL2Manager(config)

	// Test transaction submission
	tx := []byte("test transaction data")
	txHash, err := l2m.SubmitTransaction(context.Background(), tx)
	assert.NoError(t, err)
	assert.NotEmpty(t, txHash)

	// Test TPS retrieval
	tps := l2m.GetTPS()
	assert.GreaterOrEqual(t, tps, 0.0)
}

func TestBlockchainConsensus(t *testing.T) {
	config := &consensus.ConsensusConfig{
		MinValidators:  1000,
		ValidatorStake: big.NewInt(1000000),
		SlashingRate:   0.1,
		BlockTime:      time.Second * 3,
		EpochDuration:  time.Hour * 24,
	}

	bc := consensus.NewBlockchainConsensus(config)

	validator := common.HexToAddress("0x1234567890123456789012345678901234567890")

	// Test validator registration
	err := bc.RegisterValidator(context.Background(), validator, big.NewInt(1500000))
	assert.NoError(t, err)

	// Test validator count
	count := bc.GetValidatorCount()
	assert.Equal(t, 1, count)
}

func TestContractSecurity(t *testing.T) {
	config := &security.SecurityConfig{
		EnableFormalVerification: true,
		EnableStaticAnalysis:     true,
		EnableFuzzing:            true,
		EnableEmergencyPause:     true,
		AuditRequired:            true,
	}

	cs := security.NewContractSecurity(config)

	contractAddr := "0x1234567890123456789012345678901234567890"

	// Test contract audit
	audit, err := cs.AuditContract(context.Background(), contractAddr)
	assert.NoError(t, err)
	assert.NotNil(t, audit)
	assert.True(t, audit.Passed)

	// Test emergency pause
	err = cs.EmergencyPause(context.Background(), contractAddr)
	assert.NoError(t, err)

	paused := cs.IsContractPaused(contractAddr)
	assert.True(t, paused)
}

func TestBlockchainMetrics(t *testing.T) {
	bm := metrics.NewBlockchainMetrics()

	// Test transaction recording
	bm.RecordTransaction(true, 100000, time.Second*2)
	assert.Equal(t, uint64(1), bm.TotalTransactions)
	assert.Equal(t, uint64(1), bm.SuccessfulTxs)

	// Test TPS update
	bm.UpdateTPS(10000.0)
	assert.Equal(t, 10000.0, bm.AverageTPS)
	assert.Equal(t, 10000.0, bm.PeakTPS)

	// Test snapshot
	snapshot := bm.GetSnapshot()
	assert.NotNil(t, snapshot)
	assert.Equal(t, uint64(1), snapshot["total_transactions"])
	assert.Equal(t, 10000.0, snapshot["average_tps"])
}

// Helper function for GovernanceManager testing
func (gm *GovernanceManager) SetTokenBalance(address common.Address, balance *big.Int) {
	gm.mu.Lock()
	defer gm.mu.Unlock()
	gm.tokenHolders[address] = balance
}

// Benchmark tests
func BenchmarkStateRecording(b *testing.B) {
	config := &state.StateConfig{
		RPCEndpoint:   "http://localhost:8545",
		PrivateKey:    "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
		IPFSEndpoint:  "http://localhost:5001",
		CacheSize:     10000,
		BatchSize:     1000,
		SyncInterval:  time.Minute,
		GasPriceLimit: big.NewInt(50000000000),
	}

	b.Skip("Requires blockchain connection")

	sm, _ := state.NewStateManager(config)
	defer sm.Close()

	vmState := &state.VMStateRecord{
		VMID:             "bench-vm",
		Owner:            common.HexToAddress("0x1234567890123456789012345678901234567890"),
		Region:           "us-east-1",
		State:            2,
		CPUAllocation:    4,
		MemoryAllocation: 8192,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sm.RecordVMState(context.Background(), vmState)
	}
}

func BenchmarkTokenTransfer(b *testing.B) {
	config := &tokens.TokenConfig{
		EnableTokens:       true,
		InitialSupply:      big.NewInt(10000000),
		MarketplaceEnabled: false,
	}

	tm := tokens.NewTokenManager(config)

	sender := common.HexToAddress("0x1234567890123456789012345678901234567890")
	recipient := common.HexToAddress("0x0987654321098765432109876543210987654321")

	tm.MintTokens(context.Background(), sender, tokens.ResourceCPU, big.NewInt(10000000))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tm.TransferTokens(context.Background(), sender, recipient, tokens.ResourceCPU, big.NewInt(1))
	}
}
