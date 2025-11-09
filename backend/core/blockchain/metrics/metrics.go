package metrics

import (
	"math/big"
	"sync"
	"time"
)

// BlockchainMetrics tracks comprehensive blockchain metrics
type BlockchainMetrics struct {
	// Transaction metrics
	TotalTransactions   uint64
	SuccessfulTxs       uint64
	FailedTxs           uint64
	AverageTPS          float64
	PeakTPS             float64

	// Performance metrics
	AverageFinalityTime time.Duration
	AverageGasUsed      uint64
	TotalGasCost        *big.Int
	GasPriceGwei        uint64

	// Validator metrics
	TotalValidators     int
	ActiveValidators    int
	SlashedValidators   int

	// Token metrics
	TokenPriceUSD       map[string]float64
	TotalTokenSupply    map[string]*big.Int
	MarketLiquidity     map[string]*big.Int

	// Governance metrics
	TotalProposals      uint64
	ActiveProposals     uint64
	ExecutedProposals   uint64
	DAOParticipation    float64

	// Cross-chain metrics
	TotalBridgeTransfers uint64
	PendingTransfers     uint64

	// L2 metrics
	L2TPS               float64
	L2Latency           time.Duration
	TotalL2Batches      uint64

	// System metrics
	BlockchainOverhead  float64
	StateStorageSize    uint64
	IPFSStorageSize     uint64

	mu sync.RWMutex
}

// NewBlockchainMetrics creates a new blockchain metrics tracker
func NewBlockchainMetrics() *BlockchainMetrics {
	return &BlockchainMetrics{
		TotalGasCost:     big.NewInt(0),
		TokenPriceUSD:    make(map[string]float64),
		TotalTokenSupply: make(map[string]*big.Int),
		MarketLiquidity:  make(map[string]*big.Int),
	}
}

// RecordTransaction records a transaction
func (bm *BlockchainMetrics) RecordTransaction(success bool, gasUsed uint64, finalityTime time.Duration) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	bm.TotalTransactions++
	if success {
		bm.SuccessfulTxs++
	} else {
		bm.FailedTxs++
	}

	// Update average gas used
	bm.AverageGasUsed = (bm.AverageGasUsed*uint64(bm.TotalTransactions-1) + gasUsed) / uint64(bm.TotalTransactions)

	// Update average finality time
	bm.AverageFinalityTime = (bm.AverageFinalityTime*time.Duration(bm.TotalTransactions-1) + finalityTime) / time.Duration(bm.TotalTransactions)
}

// UpdateTPS updates transactions per second
func (bm *BlockchainMetrics) UpdateTPS(tps float64) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	bm.AverageTPS = tps
	if tps > bm.PeakTPS {
		bm.PeakTPS = tps
	}
}

// GetSnapshot returns a snapshot of current metrics
func (bm *BlockchainMetrics) GetSnapshot() map[string]interface{} {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	return map[string]interface{}{
		"total_transactions":      bm.TotalTransactions,
		"successful_txs":          bm.SuccessfulTxs,
		"failed_txs":              bm.FailedTxs,
		"average_tps":             bm.AverageTPS,
		"peak_tps":                bm.PeakTPS,
		"average_finality_time":   bm.AverageFinalityTime.Seconds(),
		"average_gas_used":        bm.AverageGasUsed,
		"total_validators":        bm.TotalValidators,
		"active_validators":       bm.ActiveValidators,
		"dao_participation":       bm.DAOParticipation,
		"l2_tps":                  bm.L2TPS,
		"blockchain_overhead":     bm.BlockchainOverhead,
	}
}
