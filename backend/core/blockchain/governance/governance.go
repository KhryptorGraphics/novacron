package governance

import (
	"context"
	"fmt"
	"math/big"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

// GovernanceManager manages DAO governance
type GovernanceManager struct {
	config     *GovernanceConfig
	proposals  map[string]*Proposal
	votes      map[string]map[common.Address]*Vote
	delegates  map[common.Address]common.Address
	tokenHolders map[common.Address]*big.Int
	mu         sync.RWMutex
}

// GovernanceConfig defines governance configuration
type GovernanceConfig struct {
	GovernanceToken    common.Address
	ProposalThreshold  *big.Int      // Minimum tokens to create proposal
	VotingPeriod       time.Duration  // Duration of voting
	QuorumPercentage   float64        // Minimum participation
	ExecutionDelay     time.Duration  // Time-lock for execution
	EnableQuadratic    bool           // Quadratic voting
	EnableDelegation   bool           // Liquid democracy
}

// Proposal represents a governance proposal
type Proposal struct {
	ID             string
	Proposer       common.Address
	Title          string
	Description    string
	Actions        []ProposalAction
	CreatedAt      time.Time
	VotingStarts   time.Time
	VotingEnds     time.Time
	ExecutionTime  time.Time
	State          ProposalState
	VotesFor       *big.Int
	VotesAgainst   *big.Int
	VotesAbstain   *big.Int
	Executed       bool
	Canceled       bool
}

// ProposalAction defines an action to execute if proposal passes
type ProposalAction struct {
	Target     common.Address
	Value      *big.Int
	Signature  string
	Calldata   []byte
	Description string
}

// ProposalState defines proposal states
type ProposalState string

const (
	ProposalStatePending   ProposalState = "pending"
	ProposalStateActive    ProposalState = "active"
	ProposalStateSucceeded ProposalState = "succeeded"
	ProposalStateFailed    ProposalState = "failed"
	ProposalStateQueued    ProposalState = "queued"
	ProposalStateExecuted  ProposalState = "executed"
	ProposalStateCanceled  ProposalState = "canceled"
)

// Vote represents a vote on a proposal
type Vote struct {
	Voter       common.Address
	Support     VoteType
	VotingPower *big.Int
	Reason      string
	Timestamp   time.Time
}

// VoteType defines vote types
type VoteType string

const (
	VoteTypeFor     VoteType = "for"
	VoteTypeAgainst VoteType = "against"
	VoteTypeAbstain VoteType = "abstain"
)

// NewGovernanceManager creates a new governance manager
func NewGovernanceManager(config *GovernanceConfig) *GovernanceManager {
	gm := &GovernanceManager{
		config:       config,
		proposals:    make(map[string]*Proposal),
		votes:        make(map[string]map[common.Address]*Vote),
		delegates:    make(map[common.Address]common.Address),
		tokenHolders: make(map[common.Address]*big.Int),
	}

	// Start background workers
	go gm.updateProposalStates()
	go gm.executeQueuedProposals()

	return gm
}

// CreateProposal creates a new governance proposal
func (gm *GovernanceManager) CreateProposal(ctx context.Context, proposer common.Address, title string, description string, actions []ProposalAction) (*Proposal, error) {
	// Check proposer has enough tokens
	balance := gm.getVotingPower(proposer)
	if balance.Cmp(gm.config.ProposalThreshold) < 0 {
		return nil, fmt.Errorf("insufficient tokens for proposal: required %s, has %s", gm.config.ProposalThreshold, balance)
	}

	// Generate proposal ID
	proposalID := gm.generateProposalID(proposer, title)

	// Create proposal
	proposal := &Proposal{
		ID:           proposalID,
		Proposer:     proposer,
		Title:        title,
		Description:  description,
		Actions:      actions,
		CreatedAt:    time.Now(),
		VotingStarts: time.Now().Add(time.Hour * 24), // 1 day delay
		VotingEnds:   time.Now().Add(time.Hour*24 + gm.config.VotingPeriod),
		ExecutionTime: time.Now().Add(time.Hour*24 + gm.config.VotingPeriod + gm.config.ExecutionDelay),
		State:        ProposalStatePending,
		VotesFor:     big.NewInt(0),
		VotesAgainst: big.NewInt(0),
		VotesAbstain: big.NewInt(0),
		Executed:     false,
		Canceled:     false,
	}

	gm.mu.Lock()
	gm.proposals[proposalID] = proposal
	gm.votes[proposalID] = make(map[common.Address]*Vote)
	gm.mu.Unlock()

	return proposal, nil
}

// CastVote casts a vote on a proposal
func (gm *GovernanceManager) CastVote(ctx context.Context, proposalID string, voter common.Address, support VoteType, reason string) error {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	proposal, exists := gm.proposals[proposalID]
	if !exists {
		return fmt.Errorf("proposal not found: %s", proposalID)
	}

	if proposal.State != ProposalStateActive {
		return fmt.Errorf("proposal not in voting period")
	}

	// Check if already voted
	if _, voted := gm.votes[proposalID][voter]; voted {
		return fmt.Errorf("already voted")
	}

	// Get voting power (with delegation)
	votingPower := gm.getVotingPowerWithDelegation(voter)

	if gm.config.EnableQuadratic {
		// Quadratic voting: voting power = sqrt(tokens)
		votingPower = new(big.Int).Sqrt(votingPower)
	}

	// Record vote
	vote := &Vote{
		Voter:       voter,
		Support:     support,
		VotingPower: votingPower,
		Reason:      reason,
		Timestamp:   time.Now(),
	}

	gm.votes[proposalID][voter] = vote

	// Update proposal vote counts
	switch support {
	case VoteTypeFor:
		proposal.VotesFor.Add(proposal.VotesFor, votingPower)
	case VoteTypeAgainst:
		proposal.VotesAgainst.Add(proposal.VotesAgainst, votingPower)
	case VoteTypeAbstain:
		proposal.VotesAbstain.Add(proposal.VotesAbstain, votingPower)
	}

	return nil
}

// DelegateVote delegates voting power to another address
func (gm *GovernanceManager) DelegateVote(ctx context.Context, delegator common.Address, delegatee common.Address) error {
	if !gm.config.EnableDelegation {
		return fmt.Errorf("delegation not enabled")
	}

	gm.mu.Lock()
	defer gm.mu.Unlock()

	// Check for circular delegation
	if gm.hasCircularDelegation(delegator, delegatee) {
		return fmt.Errorf("circular delegation detected")
	}

	gm.delegates[delegator] = delegatee

	return nil
}

// ExecuteProposal executes a passed proposal
func (gm *GovernanceManager) ExecuteProposal(ctx context.Context, proposalID string) error {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	proposal, exists := gm.proposals[proposalID]
	if !exists {
		return fmt.Errorf("proposal not found: %s", proposalID)
	}

	if proposal.State != ProposalStateQueued {
		return fmt.Errorf("proposal not ready for execution")
	}

	if time.Now().Before(proposal.ExecutionTime) {
		return fmt.Errorf("execution time not reached")
	}

	// Execute each action
	for _, action := range proposal.Actions {
		if err := gm.executeAction(action); err != nil {
			return fmt.Errorf("action execution failed: %w", err)
		}
	}

	proposal.State = ProposalStateExecuted
	proposal.Executed = true

	return nil
}

// CancelProposal cancels a proposal
func (gm *GovernanceManager) CancelProposal(ctx context.Context, proposalID string, canceller common.Address) error {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	proposal, exists := gm.proposals[proposalID]
	if !exists {
		return fmt.Errorf("proposal not found: %s", proposalID)
	}

	// Only proposer can cancel
	if proposal.Proposer != canceller {
		return fmt.Errorf("only proposer can cancel")
	}

	if proposal.Executed {
		return fmt.Errorf("cannot cancel executed proposal")
	}

	proposal.State = ProposalStateCanceled
	proposal.Canceled = true

	return nil
}

// getVotingPower returns voting power for an address
func (gm *GovernanceManager) getVotingPower(address common.Address) *big.Int {
	if balance, exists := gm.tokenHolders[address]; exists {
		return new(big.Int).Set(balance)
	}
	return big.NewInt(0)
}

// getVotingPowerWithDelegation returns voting power including delegated votes
func (gm *GovernanceManager) getVotingPowerWithDelegation(address common.Address) *big.Int {
	power := gm.getVotingPower(address)

	if !gm.config.EnableDelegation {
		return power
	}

	// Add delegated power
	for delegator, delegatee := range gm.delegates {
		if delegatee == address {
			delegatedPower := gm.getVotingPower(delegator)
			power.Add(power, delegatedPower)
		}
	}

	return power
}

// hasCircularDelegation checks for circular delegation
func (gm *GovernanceManager) hasCircularDelegation(delegator common.Address, delegatee common.Address) bool {
	visited := make(map[common.Address]bool)
	current := delegatee

	for {
		if current == delegator {
			return true
		}

		if visited[current] {
			return false
		}

		visited[current] = true

		next, exists := gm.delegates[current]
		if !exists {
			return false
		}

		current = next
	}
}

// executeAction executes a proposal action
func (gm *GovernanceManager) executeAction(action ProposalAction) error {
	// In production, this would call the target contract
	// For now, just log the action
	fmt.Printf("Executing action: %s on %s with value %s\n", action.Description, action.Target.Hex(), action.Value)
	return nil
}

// updateProposalStates updates proposal states based on time and votes
func (gm *GovernanceManager) updateProposalStates() {
	ticker := time.NewTicker(time.Minute * 5)
	defer ticker.Stop()

	for range ticker.C {
		gm.mu.Lock()

		for _, proposal := range gm.proposals {
			now := time.Now()

			switch proposal.State {
			case ProposalStatePending:
				if now.After(proposal.VotingStarts) {
					proposal.State = ProposalStateActive
				}

			case ProposalStateActive:
				if now.After(proposal.VotingEnds) {
					// Calculate results
					totalVotes := new(big.Int).Add(proposal.VotesFor, proposal.VotesAgainst)
					totalVotes.Add(totalVotes, proposal.VotesAbstain)

					// Check quorum
					totalSupply := gm.getTotalSupply()
					quorumRequired := new(big.Int).Div(
						new(big.Int).Mul(totalSupply, big.NewInt(int64(gm.config.QuorumPercentage*100))),
						big.NewInt(100),
					)

					if totalVotes.Cmp(quorumRequired) < 0 {
						proposal.State = ProposalStateFailed
					} else if proposal.VotesFor.Cmp(proposal.VotesAgainst) > 0 {
						proposal.State = ProposalStateSucceeded
						// Queue for execution
						proposal.State = ProposalStateQueued
					} else {
						proposal.State = ProposalStateFailed
					}
				}
			}
		}

		gm.mu.Unlock()
	}
}

// executeQueuedProposals executes queued proposals when ready
func (gm *GovernanceManager) executeQueuedProposals() {
	ticker := time.NewTicker(time.Minute * 10)
	defer ticker.Stop()

	for range ticker.C {
		gm.mu.RLock()
		queuedProposals := make([]string, 0)
		for id, proposal := range gm.proposals {
			if proposal.State == ProposalStateQueued && time.Now().After(proposal.ExecutionTime) {
				queuedProposals = append(queuedProposals, id)
			}
		}
		gm.mu.RUnlock()

		// Execute queued proposals
		for _, id := range queuedProposals {
			gm.ExecuteProposal(context.Background(), id)
		}
	}
}

// getTotalSupply returns total governance token supply
func (gm *GovernanceManager) getTotalSupply() *big.Int {
	total := big.NewInt(0)
	for _, balance := range gm.tokenHolders {
		total.Add(total, balance)
	}
	return total
}

// generateProposalID generates a unique proposal ID
func (gm *GovernanceManager) generateProposalID(proposer common.Address, title string) string {
	data := fmt.Sprintf("%s:%s:%d", proposer.Hex(), title, time.Now().Unix())
	hash := crypto.Keccak256Hash([]byte(data))
	return hash.Hex()
}

// GetProposal returns a proposal by ID
func (gm *GovernanceManager) GetProposal(proposalID string) (*Proposal, error) {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	proposal, exists := gm.proposals[proposalID]
	if !exists {
		return nil, fmt.Errorf("proposal not found: %s", proposalID)
	}

	return proposal, nil
}

// GetVote returns a vote on a proposal
func (gm *GovernanceManager) GetVote(proposalID string, voter common.Address) (*Vote, error) {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	votes, exists := gm.votes[proposalID]
	if !exists {
		return nil, fmt.Errorf("proposal not found: %s", proposalID)
	}

	vote, voted := votes[voter]
	if !voted {
		return nil, fmt.Errorf("voter has not voted")
	}

	return vote, nil
}

// ListProposals returns all proposals
func (gm *GovernanceManager) ListProposals() []*Proposal {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	proposals := make([]*Proposal, 0, len(gm.proposals))
	for _, proposal := range gm.proposals {
		proposals = append(proposals, proposal)
	}

	return proposals
}
