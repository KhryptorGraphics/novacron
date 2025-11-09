package tokens

import (
	"context"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

// TokenManager manages tokenized resources (ERC-20)
type TokenManager struct {
	config    *TokenConfig
	tokens    map[ResourceType]*TokenInfo
	balances  map[common.Address]map[ResourceType]*big.Int
	staking   map[common.Address]map[ResourceType]*StakeInfo
	market    *AutomatedMarketMaker
	mu        sync.RWMutex
}

// TokenConfig defines token manager configuration
type TokenConfig struct {
	EnableTokens       bool
	TokenAddresses     map[ResourceType]common.Address
	InitialSupply      *big.Int
	StakingEnabled     bool
	StakingRewardRate  float64  // Annual percentage rate
	MarketplaceEnabled bool
}

// ResourceType defines resource types
type ResourceType string

const (
	ResourceCPU     ResourceType = "CPU"
	ResourceMemory  ResourceType = "MEM"
	ResourceStorage ResourceType = "STO"
	ResourceNetwork ResourceType = "NET"
)

// TokenInfo represents token information
type TokenInfo struct {
	Symbol       string
	Name         string
	Address      common.Address
	TotalSupply  *big.Int
	Decimals     uint8
	PriceUSD     float64
}

// StakeInfo represents staking information
type StakeInfo struct {
	Amount       *big.Int
	StakedAt     time.Time
	UnlocksAt    time.Time
	RewardsClaim *big.Int
}

// AutomatedMarketMaker implements AMM for resource trading
type AutomatedMarketMaker struct {
	pools     map[ResourceType]*LiquidityPool
	fees      float64  // 0.3% trading fee
	mu        sync.RWMutex
}

// LiquidityPool represents a liquidity pool
type LiquidityPool struct {
	TokenReserve *big.Int
	ETHReserve   *big.Int
	TotalShares  *big.Int
	Providers    map[common.Address]*big.Int
}

// NewTokenManager creates a new token manager
func NewTokenManager(config *TokenConfig) *TokenManager {
	tm := &TokenManager{
		config:   config,
		tokens:   make(map[ResourceType]*TokenInfo),
		balances: make(map[common.Address]map[ResourceType]*big.Int),
		staking:  make(map[common.Address]map[ResourceType]*StakeInfo),
		market: &AutomatedMarketMaker{
			pools: make(map[ResourceType]*LiquidityPool),
			fees:  0.003,  // 0.3%
		},
	}

	// Initialize tokens
	tm.initializeTokens()

	// Initialize liquidity pools
	if config.MarketplaceEnabled {
		tm.initializePools()
	}

	// Start staking rewards distribution
	if config.StakingEnabled {
		go tm.distributeStakingRewards()
	}

	return tm
}

// initializeTokens initializes ERC-20 tokens for each resource type
func (tm *TokenManager) initializeTokens() {
	tokens := map[ResourceType]*TokenInfo{
		ResourceCPU: {
			Symbol:      "CPU",
			Name:        "NovaCron CPU Token",
			TotalSupply: new(big.Int).Set(tm.config.InitialSupply),
			Decimals:    18,
			PriceUSD:    0.001,
		},
		ResourceMemory: {
			Symbol:      "MEM",
			Name:        "NovaCron Memory Token",
			TotalSupply: new(big.Int).Set(tm.config.InitialSupply),
			Decimals:    18,
			PriceUSD:    0.0005,
		},
		ResourceStorage: {
			Symbol:      "STO",
			Name:        "NovaCron Storage Token",
			TotalSupply: new(big.Int).Set(tm.config.InitialSupply),
			Decimals:    18,
			PriceUSD:    0.0001,
		},
		ResourceNetwork: {
			Symbol:      "NET",
			Name:        "NovaCron Network Token",
			TotalSupply: new(big.Int).Set(tm.config.InitialSupply),
			Decimals:    18,
			PriceUSD:    0.00001,
		},
	}

	for resourceType, info := range tokens {
		if address, ok := tm.config.TokenAddresses[resourceType]; ok {
			info.Address = address
		}
		tm.tokens[resourceType] = info
	}
}

// initializePools initializes liquidity pools
func (tm *TokenManager) initializePools() {
	for resourceType := range tm.tokens {
		tm.market.pools[resourceType] = &LiquidityPool{
			TokenReserve: big.NewInt(0),
			ETHReserve:   big.NewInt(0),
			TotalShares:  big.NewInt(0),
			Providers:    make(map[common.Address]*big.Int),
		}
	}
}

// MintTokens mints new resource tokens
func (tm *TokenManager) MintTokens(ctx context.Context, to common.Address, resourceType ResourceType, amount *big.Int) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if _, exists := tm.balances[to]; !exists {
		tm.balances[to] = make(map[ResourceType]*big.Int)
	}

	if _, exists := tm.balances[to][resourceType]; !exists {
		tm.balances[to][resourceType] = big.NewInt(0)
	}

	tm.balances[to][resourceType].Add(tm.balances[to][resourceType], amount)
	tm.tokens[resourceType].TotalSupply.Add(tm.tokens[resourceType].TotalSupply, amount)

	return nil
}

// TransferTokens transfers tokens between addresses
func (tm *TokenManager) TransferTokens(ctx context.Context, from common.Address, to common.Address, resourceType ResourceType, amount *big.Int) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Check balance
	if tm.balances[from][resourceType].Cmp(amount) < 0 {
		return fmt.Errorf("insufficient balance")
	}

	// Deduct from sender
	tm.balances[from][resourceType].Sub(tm.balances[from][resourceType], amount)

	// Add to receiver
	if _, exists := tm.balances[to]; !exists {
		tm.balances[to] = make(map[ResourceType]*big.Int)
	}
	if _, exists := tm.balances[to][resourceType]; !exists {
		tm.balances[to][resourceType] = big.NewInt(0)
	}
	tm.balances[to][resourceType].Add(tm.balances[to][resourceType], amount)

	return nil
}

// StakeTokens stakes tokens for guaranteed resources
func (tm *TokenManager) StakeTokens(ctx context.Context, user common.Address, resourceType ResourceType, amount *big.Int, duration time.Duration) error {
	if !tm.config.StakingEnabled {
		return fmt.Errorf("staking not enabled")
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Check balance
	if tm.balances[user][resourceType].Cmp(amount) < 0 {
		return fmt.Errorf("insufficient balance")
	}

	// Deduct from balance
	tm.balances[user][resourceType].Sub(tm.balances[user][resourceType], amount)

	// Create stake
	if _, exists := tm.staking[user]; !exists {
		tm.staking[user] = make(map[ResourceType]*StakeInfo)
	}

	tm.staking[user][resourceType] = &StakeInfo{
		Amount:       amount,
		StakedAt:     time.Now(),
		UnlocksAt:    time.Now().Add(duration),
		RewardsClaim: big.NewInt(0),
	}

	return nil
}

// UnstakeTokens unstakes tokens after lock period
func (tm *TokenManager) UnstakeTokens(ctx context.Context, user common.Address, resourceType ResourceType) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	stake, exists := tm.staking[user][resourceType]
	if !exists {
		return fmt.Errorf("no stake found")
	}

	if time.Now().Before(stake.UnlocksAt) {
		return fmt.Errorf("stake still locked")
	}

	// Return staked amount + rewards
	totalReturn := new(big.Int).Add(stake.Amount, stake.RewardsClaim)
	tm.balances[user][resourceType].Add(tm.balances[user][resourceType], totalReturn)

	// Remove stake
	delete(tm.staking[user], resourceType)

	return nil
}

// AddLiquidity adds liquidity to AMM pool
func (tm *TokenManager) AddLiquidity(ctx context.Context, provider common.Address, resourceType ResourceType, tokenAmount *big.Int, ethAmount *big.Int) error {
	if !tm.config.MarketplaceEnabled {
		return fmt.Errorf("marketplace not enabled")
	}

	tm.market.mu.Lock()
	defer tm.market.mu.Unlock()

	pool := tm.market.pools[resourceType]

	// Calculate shares
	var shares *big.Int
	if pool.TotalShares.Cmp(big.NewInt(0)) == 0 {
		// Initial liquidity
		shares = new(big.Int).Sqrt(new(big.Int).Mul(tokenAmount, ethAmount))
	} else {
		// Proportional shares
		tokenShare := new(big.Int).Div(new(big.Int).Mul(tokenAmount, pool.TotalShares), pool.TokenReserve)
		ethShare := new(big.Int).Div(new(big.Int).Mul(ethAmount, pool.TotalShares), pool.ETHReserve)
		shares = tokenShare
		if ethShare.Cmp(tokenShare) < 0 {
			shares = ethShare
		}
	}

	// Update pool reserves
	pool.TokenReserve.Add(pool.TokenReserve, tokenAmount)
	pool.ETHReserve.Add(pool.ETHReserve, ethAmount)
	pool.TotalShares.Add(pool.TotalShares, shares)

	// Update provider shares
	if _, exists := pool.Providers[provider]; !exists {
		pool.Providers[provider] = big.NewInt(0)
	}
	pool.Providers[provider].Add(pool.Providers[provider], shares)

	return nil
}

// SwapTokens swaps tokens using AMM (bonding curve)
func (tm *TokenManager) SwapTokens(ctx context.Context, buyer common.Address, resourceType ResourceType, ethIn *big.Int) (*big.Int, error) {
	if !tm.config.MarketplaceEnabled {
		return nil, fmt.Errorf("marketplace not enabled")
	}

	tm.market.mu.Lock()
	defer tm.market.mu.Unlock()

	pool := tm.market.pools[resourceType]

	// Calculate output using constant product formula (x * y = k)
	// tokenOut = (tokenReserve * ethIn * (1 - fee)) / (ethReserve + ethIn)

	fee := new(big.Int).Div(new(big.Int).Mul(ethIn, big.NewInt(3)), big.NewInt(1000)) // 0.3%
	ethInAfterFee := new(big.Int).Sub(ethIn, fee)

	numerator := new(big.Int).Mul(pool.TokenReserve, ethInAfterFee)
	denominator := new(big.Int).Add(pool.ETHReserve, ethInAfterFee)
	tokenOut := new(big.Int).Div(numerator, denominator)

	// Update reserves
	pool.TokenReserve.Sub(pool.TokenReserve, tokenOut)
	pool.ETHReserve.Add(pool.ETHReserve, ethIn)

	// Update buyer balance
	tm.mu.Lock()
	if _, exists := tm.balances[buyer]; !exists {
		tm.balances[buyer] = make(map[ResourceType]*big.Int)
	}
	if _, exists := tm.balances[buyer][resourceType]; !exists {
		tm.balances[buyer][resourceType] = big.NewInt(0)
	}
	tm.balances[buyer][resourceType].Add(tm.balances[buyer][resourceType], tokenOut)
	tm.mu.Unlock()

	return tokenOut, nil
}

// GetSpotPrice returns current spot price for resource
func (tm *TokenManager) GetSpotPrice(resourceType ResourceType) (*big.Int, error) {
	if !tm.config.MarketplaceEnabled {
		return nil, fmt.Errorf("marketplace not enabled")
	}

	tm.market.mu.RLock()
	defer tm.market.mu.RUnlock()

	pool := tm.market.pools[resourceType]

	if pool.TokenReserve.Cmp(big.NewInt(0)) == 0 {
		return big.NewInt(0), nil
	}

	// Price = ETHReserve / TokenReserve
	price := new(big.Int).Div(pool.ETHReserve, pool.TokenReserve)

	return price, nil
}

// distributeStakingRewards distributes staking rewards periodically
func (tm *TokenManager) distributeStakingRewards() {
	ticker := time.NewTicker(time.Hour * 24) // Daily rewards
	defer ticker.Stop()

	for range ticker.C {
		tm.mu.Lock()

		for user, stakes := range tm.staking {
			for resourceType, stake := range stakes {
				// Calculate daily reward (APR / 365)
				dailyRate := tm.config.StakingRewardRate / 365.0
				reward := new(big.Int).Div(
					new(big.Int).Mul(stake.Amount, big.NewInt(int64(dailyRate*1e18))),
					big.NewInt(1e18),
				)

				stake.RewardsClaim.Add(stake.RewardsClaim, reward)
			}
		}

		tm.mu.Unlock()
	}
}

// GetBalance returns token balance for address
func (tm *TokenManager) GetBalance(address common.Address, resourceType ResourceType) *big.Int {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	if balances, exists := tm.balances[address]; exists {
		if balance, exists := balances[resourceType]; exists {
			return new(big.Int).Set(balance)
		}
	}

	return big.NewInt(0)
}

// GetStakeInfo returns staking information
func (tm *TokenManager) GetStakeInfo(address common.Address, resourceType ResourceType) *StakeInfo {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	if stakes, exists := tm.staking[address]; exists {
		if stake, exists := stakes[resourceType]; exists {
			return &StakeInfo{
				Amount:       new(big.Int).Set(stake.Amount),
				StakedAt:     stake.StakedAt,
				UnlocksAt:    stake.UnlocksAt,
				RewardsClaim: new(big.Int).Set(stake.RewardsClaim),
			}
		}
	}

	return nil
}

// GetTokenInfo returns token information
func (tm *TokenManager) GetTokenInfo(resourceType ResourceType) *TokenInfo {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	if info, exists := tm.tokens[resourceType]; exists {
		return &TokenInfo{
			Symbol:      info.Symbol,
			Name:        info.Name,
			Address:     info.Address,
			TotalSupply: new(big.Int).Set(info.TotalSupply),
			Decimals:    info.Decimals,
			PriceUSD:    info.PriceUSD,
		}
	}

	return nil
}

// GetPoolInfo returns liquidity pool information
func (tm *TokenManager) GetPoolInfo(resourceType ResourceType) (tokenReserve *big.Int, ethReserve *big.Int, totalShares *big.Int) {
	if !tm.config.MarketplaceEnabled {
		return big.NewInt(0), big.NewInt(0), big.NewInt(0)
	}

	tm.market.mu.RLock()
	defer tm.market.mu.RUnlock()

	pool := tm.market.pools[resourceType]
	return new(big.Int).Set(pool.TokenReserve), new(big.Int).Set(pool.ETHReserve), new(big.Int).Set(pool.TotalShares)
}
