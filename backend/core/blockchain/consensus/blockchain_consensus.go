package consensus

import (
	"context"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
)

// BlockchainConsensus implements PoS consensus via blockchain
type BlockchainConsensus struct {
	config     *ConsensusConfig
	validators map[common.Address]*Validator
	blocks     []*Block
	mu         sync.RWMutex
}

// ConsensusConfig defines consensus configuration
type ConsensusConfig struct {
	MinValidators    int
	ValidatorStake   *big.Int
	SlashingRate     float64
	BlockTime        time.Duration
	EpochDuration    time.Duration
}

// Validator represents a PoS validator
type Validator struct {
	Address     common.Address
	Stake       *big.Int
	IsActive    bool
	LastBlock   uint64
	SlashCount  int
}

// Block represents a consensus block
type Block struct {
	Number    uint64
	Hash      string
	Validator common.Address
	Timestamp time.Time
}

// NewBlockchainConsensus creates a new blockchain consensus manager
func NewBlockchainConsensus(config *ConsensusConfig) *BlockchainConsensus {
	return &BlockchainConsensus{
		config:     config,
		validators: make(map[common.Address]*Validator),
		blocks:     make([]*Block, 0),
	}
}

// RegisterValidator registers a new validator
func (bc *BlockchainConsensus) RegisterValidator(ctx context.Context, validator common.Address, stake *big.Int) error {
	if stake.Cmp(bc.config.ValidatorStake) < 0 {
		return fmt.Errorf("insufficient stake")
	}

	bc.mu.Lock()
	defer bc.mu.Unlock()

	bc.validators[validator] = &Validator{
		Address:  validator,
		Stake:    stake,
		IsActive: true,
	}

	return nil
}

// GetValidatorCount returns the number of active validators
func (bc *BlockchainConsensus) GetValidatorCount() int {
	bc.mu.RLock()
	defer bc.mu.RUnlock()

	count := 0
	for _, v := range bc.validators {
		if v.IsActive {
			count++
		}
	}
	return count
}
