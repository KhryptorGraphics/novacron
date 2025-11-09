package blockchain

import (
	"context"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"

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

// BlockchainIntegration provides unified blockchain integration
type BlockchainIntegration struct {
	config              *BlockchainConfig
	stateManager        *state.StateManager
	contractOrchestrator *contracts.ContractOrchestrator
	didManager          *did.DIDManager
	tokenManager        *tokens.TokenManager
	governanceManager   *governance.GovernanceManager
	crossChainBridge    *crosschain.CrossChainBridge
	consensusManager    *consensus.BlockchainConsensus
	l2Manager           *l2.L2Manager
	securityManager     *security.ContractSecurity
	metricsManager      *metrics.BlockchainMetrics
	mu                  sync.RWMutex
}

// NewBlockchainIntegration creates a new blockchain integration
func NewBlockchainIntegration(config *BlockchainConfig) (*BlockchainIntegration, error) {
	if !config.EnableBlockchain {
		return nil, fmt.Errorf("blockchain not enabled")
	}

	bi := &BlockchainIntegration{
		config:         config,
		metricsManager: metrics.NewBlockchainMetrics(),
	}

	// Initialize state manager
	if err := bi.initStateManager(); err != nil {
		return nil, fmt.Errorf("failed to initialize state manager: %w", err)
	}

	// Initialize contract orchestrator
	if err := bi.initContractOrchestrator(); err != nil {
		return nil, fmt.Errorf("failed to initialize contract orchestrator: %w", err)
	}

	// Initialize DID manager
	if config.EnableDID {
		if err := bi.initDIDManager(); err != nil {
			return nil, fmt.Errorf("failed to initialize DID manager: %w", err)
		}
	}

	// Initialize token manager
	if config.TokenizedResources {
		if err := bi.initTokenManager(); err != nil {
			return nil, fmt.Errorf("failed to initialize token manager: %w", err)
		}
	}

	// Initialize governance
	if config.GovernanceEnabled {
		if err := bi.initGovernanceManager(); err != nil {
			return nil, fmt.Errorf("failed to initialize governance: %w", err)
		}
	}

	// Initialize cross-chain bridge
	if config.EnableCrossChain {
		if err := bi.initCrossChainBridge(); err != nil {
			return nil, fmt.Errorf("failed to initialize cross-chain bridge: %w", err)
		}
	}

	// Initialize consensus
	if err := bi.initConsensusManager(); err != nil {
		return nil, fmt.Errorf("failed to initialize consensus: %w", err)
	}

	// Initialize L2
	if config.UseLayer2 {
		if err := bi.initL2Manager(); err != nil {
			return nil, fmt.Errorf("failed to initialize L2 manager: %w", err)
		}
	}

	// Initialize security
	if err := bi.initSecurityManager(); err != nil {
		return nil, fmt.Errorf("failed to initialize security manager: %w", err)
	}

	// Start background workers
	go bi.collectMetrics()

	return bi, nil
}

// initStateManager initializes the state manager
func (bi *BlockchainIntegration) initStateManager() error {
	sm, err := state.NewStateManager(&state.StateConfig{
		RPCEndpoint:   bi.config.RPCEndpoint,
		PrivateKey:    bi.config.PrivateKey,
		IPFSEndpoint:  bi.config.IPFSEndpoint,
		CacheSize:     10000,
		BatchSize:     100,
		SyncInterval:  time.Minute * 5,
		GasPriceLimit: bi.config.GasPriceLimit,
	})
	if err != nil {
		return err
	}

	bi.stateManager = sm
	return nil
}

// initContractOrchestrator initializes the contract orchestrator
func (bi *BlockchainIntegration) initContractOrchestrator() error {
	co, err := contracts.NewContractOrchestrator(&contracts.OrchestratorConfig{
		RPCEndpoint:       bi.config.RPCEndpoint,
		PrivateKey:        bi.config.PrivateKey,
		ChainID:           big.NewInt(137), // Polygon
		GasLimit:          bi.config.MaxGasPerTx,
		GasPrice:          bi.config.GasPriceLimit,
		MultiSigThreshold: bi.config.MultiSigThreshold,
		TimeLockDelay:     time.Hour * 24,
	})
	if err != nil {
		return err
	}

	bi.contractOrchestrator = co
	return nil
}

// initDIDManager initializes the DID manager
func (bi *BlockchainIntegration) initDIDManager() error {
	bi.didManager = did.NewDIDManager(&did.DIDConfig{
		Network:          bi.config.Network,
		RegistryContract: common.HexToAddress(bi.config.DIDRegistry),
		EnableZKProofs:   true,
		CredentialTTL:    time.Hour * 24 * 365,
	})
	return nil
}

// initTokenManager initializes the token manager
func (bi *BlockchainIntegration) initTokenManager() error {
	bi.tokenManager = tokens.NewTokenManager(&tokens.TokenConfig{
		EnableTokens:       bi.config.TokenizedResources,
		TokenAddresses:     bi.config.ContractAddresses,
		InitialSupply:      big.NewInt(10000000),
		StakingEnabled:     true,
		StakingRewardRate:  0.1,
		MarketplaceEnabled: bi.config.MarketplaceEnabled,
	})
	return nil
}

// initGovernanceManager initializes the governance manager
func (bi *BlockchainIntegration) initGovernanceManager() error {
	bi.governanceManager = governance.NewGovernanceManager(&governance.GovernanceConfig{
		GovernanceToken:   common.HexToAddress(bi.config.ContractAddresses["governance"]),
		ProposalThreshold: big.NewInt(int64(bi.config.ProposalThreshold)),
		VotingPeriod:      bi.config.VotingPeriod,
		QuorumPercentage:  bi.config.QuorumPercentage,
		ExecutionDelay:    time.Hour * 48,
		EnableQuadratic:   true,
		EnableDelegation:  true,
	})
	return nil
}

// initCrossChainBridge initializes the cross-chain bridge
func (bi *BlockchainIntegration) initCrossChainBridge() error {
	bi.crossChainBridge = crosschain.NewCrossChainBridge(&crosschain.BridgeConfig{
		SupportedChains:    bi.config.SupportedChains,
		BridgeAddresses:    bi.config.BridgeAddresses,
		ValidatorThreshold: 3,
		TransferTimeout:    time.Minute * 30,
	})
	return nil
}

// initConsensusManager initializes the consensus manager
func (bi *BlockchainIntegration) initConsensusManager() error {
	bi.consensusManager = consensus.NewBlockchainConsensus(&consensus.ConsensusConfig{
		MinValidators:  bi.config.MinValidators,
		ValidatorStake: big.NewInt(int64(bi.config.ValidatorStake)),
		SlashingRate:   bi.config.SlashingRate,
		BlockTime:      time.Second * 3,
		EpochDuration:  time.Hour * 24,
	})
	return nil
}

// initL2Manager initializes the L2 manager
func (bi *BlockchainIntegration) initL2Manager() error {
	bi.l2Manager = l2.NewL2Manager(&l2.L2Config{
		EnableL2:     bi.config.UseLayer2,
		L2Type:       "optimistic",
		BatchSize:    1000,
		SequencerURL: bi.config.L2RPCEndpoint,
		TargetTPS:    10000,
	})
	return nil
}

// initSecurityManager initializes the security manager
func (bi *BlockchainIntegration) initSecurityManager() error {
	bi.securityManager = security.NewContractSecurity(&security.SecurityConfig{
		EnableFormalVerification: bi.config.RequireAudit,
		EnableStaticAnalysis:     true,
		EnableFuzzing:            true,
		EnableEmergencyPause:     bi.config.EmergencyPauseEnabled,
		AuditRequired:            bi.config.RequireAudit,
	})
	return nil
}

// collectMetrics collects blockchain metrics periodically
func (bi *BlockchainIntegration) collectMetrics() {
	ticker := time.NewTicker(bi.config.MetricsInterval)
	defer ticker.Stop()

	for range ticker.C {
		// Collect state manager metrics
		if bi.stateManager != nil {
			stateMetrics := bi.stateManager.GetMetrics()
			bi.metricsManager.UpdateTPS(stateMetrics.AverageTPS)
		}

		// Collect L2 metrics
		if bi.l2Manager != nil {
			tps := bi.l2Manager.GetTPS()
			bi.metricsManager.L2TPS = tps
		}

		// Collect consensus metrics
		if bi.consensusManager != nil {
			bi.metricsManager.ActiveValidators = bi.consensusManager.GetValidatorCount()
		}
	}
}

// GetStateManager returns the state manager
func (bi *BlockchainIntegration) GetStateManager() *state.StateManager {
	return bi.stateManager
}

// GetContractOrchestrator returns the contract orchestrator
func (bi *BlockchainIntegration) GetContractOrchestrator() *contracts.ContractOrchestrator {
	return bi.contractOrchestrator
}

// GetDIDManager returns the DID manager
func (bi *BlockchainIntegration) GetDIDManager() *did.DIDManager {
	return bi.didManager
}

// GetTokenManager returns the token manager
func (bi *BlockchainIntegration) GetTokenManager() *tokens.TokenManager {
	return bi.tokenManager
}

// GetGovernanceManager returns the governance manager
func (bi *BlockchainIntegration) GetGovernanceManager() *governance.GovernanceManager {
	return bi.governanceManager
}

// GetCrossChainBridge returns the cross-chain bridge
func (bi *BlockchainIntegration) GetCrossChainBridge() *crosschain.CrossChainBridge {
	return bi.crossChainBridge
}

// GetConsensusManager returns the consensus manager
func (bi *BlockchainIntegration) GetConsensusManager() *consensus.BlockchainConsensus {
	return bi.consensusManager
}

// GetL2Manager returns the L2 manager
func (bi *BlockchainIntegration) GetL2Manager() *l2.L2Manager {
	return bi.l2Manager
}

// GetSecurityManager returns the security manager
func (bi *BlockchainIntegration) GetSecurityManager() *security.ContractSecurity {
	return bi.securityManager
}

// GetMetrics returns blockchain metrics
func (bi *BlockchainIntegration) GetMetrics() map[string]interface{} {
	return bi.metricsManager.GetSnapshot()
}

// Close closes all blockchain connections
func (bi *BlockchainIntegration) Close() error {
	if bi.stateManager != nil {
		bi.stateManager.Close()
	}
	if bi.contractOrchestrator != nil {
		bi.contractOrchestrator.Close()
	}
	return nil
}
