package blockchain

import (
	"time"
)

// BlockchainConfig defines blockchain integration configuration
type BlockchainConfig struct {
	// Core settings
	EnableBlockchain      bool                       `json:"enable_blockchain" yaml:"enable_blockchain"`
	Network               string                     `json:"network" yaml:"network"` // "ethereum", "polygon", "solana"
	ContractAddresses     map[string]string          `json:"contract_addresses" yaml:"contract_addresses"`
	PrivateKey            string                     `json:"private_key" yaml:"private_key"`
	RPCEndpoint           string                     `json:"rpc_endpoint" yaml:"rpc_endpoint"`

	// Layer 2 settings
	UseLayer2             bool                       `json:"use_layer2" yaml:"use_layer2"`
	L2Network             string                     `json:"l2_network" yaml:"l2_network"` // "polygon", "optimism", "arbitrum", "zksync"
	L2RPCEndpoint         string                     `json:"l2_rpc_endpoint" yaml:"l2_rpc_endpoint"`

	// Gas optimization
	GasPriceLimit         uint64                     `json:"gas_price_limit" yaml:"gas_price_limit"` // wei
	MaxGasPerTx           uint64                     `json:"max_gas_per_tx" yaml:"max_gas_per_tx"`
	GasStrategy           string                     `json:"gas_strategy" yaml:"gas_strategy"` // "slow", "medium", "fast"

	// Consensus settings
	MinValidators         int                        `json:"min_validators" yaml:"min_validators"` // 1000
	ValidatorStake        uint64                     `json:"validator_stake" yaml:"validator_stake"`
	SlashingRate          float64                    `json:"slashing_rate" yaml:"slashing_rate"`
	ConsensusTimeout      time.Duration              `json:"consensus_timeout" yaml:"consensus_timeout"`

	// Governance
	GovernanceEnabled     bool                       `json:"governance_enabled" yaml:"governance_enabled"`
	ProposalThreshold     uint64                     `json:"proposal_threshold" yaml:"proposal_threshold"`
	VotingPeriod          time.Duration              `json:"voting_period" yaml:"voting_period"`
	QuorumPercentage      float64                    `json:"quorum_percentage" yaml:"quorum_percentage"`

	// Tokenized resources
	TokenizedResources    bool                       `json:"tokenized_resources" yaml:"tokenized_resources"`
	TokenPrices           map[string]float64         `json:"token_prices" yaml:"token_prices"`
	MarketplaceEnabled    bool                       `json:"marketplace_enabled" yaml:"marketplace_enabled"`

	// DID settings
	EnableDID             bool                       `json:"enable_did" yaml:"enable_did"`
	DIDRegistry           string                     `json:"did_registry" yaml:"did_registry"`

	// Cross-chain
	EnableCrossChain      bool                       `json:"enable_cross_chain" yaml:"enable_cross_chain"`
	SupportedChains       []string                   `json:"supported_chains" yaml:"supported_chains"`
	BridgeAddresses       map[string]string          `json:"bridge_addresses" yaml:"bridge_addresses"`

	// IPFS settings
	IPFSEnabled           bool                       `json:"ipfs_enabled" yaml:"ipfs_enabled"`
	IPFSEndpoint          string                     `json:"ipfs_endpoint" yaml:"ipfs_endpoint"`

	// Security
	MultiSigThreshold     int                        `json:"multisig_threshold" yaml:"multisig_threshold"` // 2-of-3, 3-of-5
	RequireAudit          bool                       `json:"require_audit" yaml:"require_audit"`
	EmergencyPauseEnabled bool                       `json:"emergency_pause_enabled" yaml:"emergency_pause_enabled"`

	// Monitoring
	MetricsEnabled        bool                       `json:"metrics_enabled" yaml:"metrics_enabled"`
	MetricsInterval       time.Duration              `json:"metrics_interval" yaml:"metrics_interval"`
}

// DefaultBlockchainConfig returns default configuration
func DefaultBlockchainConfig() *BlockchainConfig {
	return &BlockchainConfig{
		EnableBlockchain:      true,
		Network:               "polygon",
		UseLayer2:             true,
		L2Network:             "polygon",
		GasPriceLimit:         50000000000, // 50 gwei
		MaxGasPerTx:           1000000,
		GasStrategy:           "medium",
		MinValidators:         1000,
		ValidatorStake:        1000000, // 1M tokens
		SlashingRate:          0.1,     // 10% slash
		ConsensusTimeout:      time.Second * 30,
		GovernanceEnabled:     true,
		ProposalThreshold:     100000, // 100k tokens
		VotingPeriod:          time.Hour * 24 * 7, // 1 week
		QuorumPercentage:      0.4,    // 40% quorum
		TokenizedResources:    true,
		TokenPrices: map[string]float64{
			"CPU": 0.001, // $0.001 per vCPU hour
			"MEM": 0.0005, // $0.0005 per GB hour
			"STO": 0.0001, // $0.0001 per TB hour
			"NET": 0.00001, // $0.00001 per GB transferred
		},
		MarketplaceEnabled:    true,
		EnableDID:             true,
		EnableCrossChain:      true,
		SupportedChains:       []string{"ethereum", "polygon", "solana", "cosmos", "avalanche"},
		IPFSEnabled:           true,
		IPFSEndpoint:          "https://ipfs.infura.io:5001",
		MultiSigThreshold:     3, // 3-of-5
		RequireAudit:          true,
		EmergencyPauseEnabled: true,
		MetricsEnabled:        true,
		MetricsInterval:       time.Second * 30,
		ContractAddresses:     make(map[string]string),
		BridgeAddresses:       make(map[string]string),
	}
}

// ContractType defines types of smart contracts
type ContractType string

const (
	ContractVMRegistry    ContractType = "vm_registry"
	ContractVMLifecycle   ContractType = "vm_lifecycle"
	ContractResourceMarket ContractType = "resource_market"
	ContractSLAContract   ContractType = "sla_contract"
	ContractGovernance    ContractType = "governance"
	ContractDIDRegistry   ContractType = "did_registry"
	ContractTokenCPU      ContractType = "token_cpu"
	ContractTokenMEM      ContractType = "token_mem"
	ContractTokenSTO      ContractType = "token_sto"
	ContractTokenNET      ContractType = "token_net"
)

// VMState defines blockchain VM states
type VMState uint8

const (
	VMStateStopped VMState = iota
	VMStateStarting
	VMStateRunning
	VMStateMigrating
	VMStatePaused
	VMStateDestroyed
)

func (s VMState) String() string {
	switch s {
	case VMStateStopped:
		return "stopped"
	case VMStateStarting:
		return "starting"
	case VMStateRunning:
		return "running"
	case VMStateMigrating:
		return "migrating"
	case VMStatePaused:
		return "paused"
	case VMStateDestroyed:
		return "destroyed"
	default:
		return "unknown"
	}
}

// ProposalState defines governance proposal states
type ProposalState uint8

const (
	ProposalStatePending ProposalState = iota
	ProposalStateActive
	ProposalStateSucceeded
	ProposalStateFailed
	ProposalStateExecuted
	ProposalStateCanceled
)

func (s ProposalState) String() string {
	switch s {
	case ProposalStatePending:
		return "pending"
	case ProposalStateActive:
		return "active"
	case ProposalStateSucceeded:
		return "succeeded"
	case ProposalStateFailed:
		return "failed"
	case ProposalStateExecuted:
		return "executed"
	case ProposalStateCanceled:
		return "canceled"
	default:
		return "unknown"
	}
}
