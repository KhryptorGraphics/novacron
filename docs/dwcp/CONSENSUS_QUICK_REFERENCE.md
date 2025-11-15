# Byzantine Consensus Quick Reference

**Location**: `/backend/core/network/dwcp/v3/consensus/`

---

## Protocol Selection Matrix

| Network Type | Protocol | Latency | Throughput | Byzantine Tolerance | Key Feature |
|-------------|----------|---------|------------|-------------------|-------------|
| **Datacenter** | Raft | <100ms | Very High | Crash-only | Fast, simple |
| **WAN** | ProBFT | <1s | High | 33% | VRF leader election |
| **Internet** | PBFT | 1-5s | Medium | 33% | Classical BFT |
| **Trust-Aware** | TP-BFT | <1s | High+26% | 33% | EigenTrust reputation |
| **High-Performance** | BullShark | 100ms | Very High | 33% | DAG-based |

---

## Byzantine Fault Tolerance Summary

### Maximum Byzantine Nodes
```
f_max = ⌊(n-1)/3⌋
Tolerance: 33.33% maximum

Examples:
- 4 nodes → 1 Byzantine (25%)
- 7 nodes → 2 Byzantine (28.5%)
- 10 nodes → 3 Byzantine (30%)
- 100 nodes → 33 Byzantine (33%)
```

### Quorum Requirements

**PBFT & TP-BFT**:
- Prepare: 2f messages
- Commit: 2f+1 messages

**ProBFT**:
- Probabilistic: √n messages
- Classical fallback: ⌈(n+f)/2⌉ + 1

**BullShark**:
- Quorum: 67% of committee

---

## Security Features

### Cryptographic Verification

**Ed25519 Signatures**:
- VRF leader election (ProBFT)
- Message authentication (all protocols)
- Public key: 32 bytes
- Signature: 64 bytes

**SHA-256 Digests**:
- Request integrity verification
- Prevents message tampering
- Output: 32 bytes

**VRF Proofs** (ProBFT only):
- Deterministic leader selection
- Unpredictable without private key
- Publicly verifiable

### Attack Mitigation

| Attack | PBFT | ProBFT | TP-BFT | BullShark |
|--------|------|--------|--------|-----------|
| Sybil | Quorum | VRF | Trust | Quorum |
| Double-Spend | Sequence# | Sequence# | Digest | DAG |
| Replay | Sequence# | Timestamp | Sequence# | Round# |
| Fork | View-change | VRF | View-change | DAG |
| Malicious Leader | View-change* | VRF rotation | Trust exclusion | Quorum |
| DoS | Rate limit** | Rate limit** | Trust decay | Buffer |

*Incomplete implementation (TODO at pbft.go:544)
**Not yet implemented

---

## Implementation Files

### PBFT (Practical Byzantine Fault Tolerance)
```
pbft.go                  (628 lines)
pbft_test.go            (314 lines)
```

**Key Functions**:
- `NewPBFT()` - Initialize consensus instance
- `Consensus()` - Execute 3-phase protocol
- `handlePrePrepare()` - Phase 1: Primary broadcast
- `handlePrepare()` - Phase 2: Replica agreement
- `handleCommit()` - Phase 3: Final commitment
- `GetMetrics()` - Performance statistics

### ProBFT (Probabilistic BFT)
```
probft/consensus.go     (478 lines)
probft/vrf.go          (174 lines)
probft/quorum.go       (214 lines)
probft/byzantine_test.go (292 lines)
```

**Key Functions**:
- `NewProBFT()` - Create consensus engine
- `ProposeBlock()` - VRF-based proposal
- `HandleMessage()` - Process consensus messages
- VRF: `Prove()`, `Verify()`
- Quorum: `CalculateQuorum()`, `IsByzantineTolerant()`

### TP-BFT (Trust-Based PBFT)
```
tpbft/tpbft.go         (446 lines)
tpbft/eigentrust.go    (286 lines)
tpbft/trust_manager.go (323 lines)
```

**Key Functions**:
- `NewTPBFT()` - Initialize with trust manager
- `SelectCommittee()` - Trust-based committee selection
- `Consensus()` - Trust-weighted voting
- EigenTrust: `ComputeGlobalTrust()`, `GetTopNodes()`
- Trust: `RecordInteraction()`, `GetTrustScore()`

### BullShark (DAG Consensus)
```
bullshark/consensus.go  (371 lines)
bullshark/dag.go       (DAG structure)
bullshark/ordering.go  (Transaction ordering)
```

**Key Functions**:
- `NewBullshark()` - Create DAG consensus
- `ProposeBlock()` - Add vertex to DAG
- `advanceRound()` - Commit transactions
- `orderAndCommit()` - Deterministic ordering

---

## Performance Characteristics

### PBFT
**Latency**: 1-5 seconds
**Throughput**: Medium
**Message Complexity**: O(n²)
**Best For**: Internet-scale untrusted networks

### ProBFT
**Latency**: <1 second
**Throughput**: High
**Message Complexity**: O(n√n)
**Best For**: Low-latency WAN deployments

### TP-BFT
**Latency**: <1 second
**Throughput**: +26% vs PBFT
**Message Complexity**: O(n²) but filtered by trust
**Best For**: Networks with reputation systems

### BullShark
**Latency**: 100ms (configurable)
**Throughput**: Very High (1000 tx/batch)
**Message Complexity**: O(n)
**Best For**: High-throughput applications

---

## Configuration Examples

### PBFT
```go
pbft, err := NewPBFT(
    "node-1",           // Node ID
    7,                  // Replica count (3f+1 = 7 for f=2)
    transport,          // Network layer
    stateMachine,       // State machine
    logger,             // Logger
)
```

### ProBFT
```go
config := QuorumConfig{
    TotalNodes:      10,
    ByzantineNodes:  3,
    ConfidenceLevel: 0.99,
}

vrf, _ := NewVRF()
probft, _ := NewProBFT("node-1", vrf, config)
```

### TP-BFT
```go
trustMgr := NewTrustManager()
tpbft := NewTPBFT("node-1", trustMgr)
tpbft.committeeSize = 10  // Committee size

committee := tpbft.SelectCommittee()  // Top trusted nodes
```

### BullShark
```go
config := DefaultConfig()
config.RoundDuration = 100 * time.Millisecond
config.QuorumThreshold = 0.67

bullshark := NewBullshark("node-1", committee, config)
bullshark.Start()
```

---

## Testing Byzantine Scenarios

### Example: 33% Byzantine Nodes
```go
totalNodes := 10
byzantineNodes := 3  // 30% Byzantine

// Test tolerance
isTolerant := IsByzantineTolerant(totalNodes, byzantineNodes)
// Result: true (10 >= 3*3+1 = 10)

// Calculate quorum
quorum := CalculatePBFTQuorum(byzantineNodes)
// Result: 7 (2*3+1)
```

### Example: VRF Leader Election
```go
vrf, _ := NewVRF()
input := []byte("round-1-view-0")

// Generate proof
proof, _ := vrf.Prove(input)

// Select leader
leaderIndex := SelectLeader(proof.Output, 10)
// Result: Deterministic index 0-9

// Verify proof
valid := vrf.Verify(input, proof)
// Result: true
```

### Example: Trust Score Management
```go
trustMgr := NewTrustManager()

// Record successful interaction
trustMgr.RecordInteraction(
    "node-1", "node-2",
    CorrectVote,
    "Correct consensus vote"
)

// Detect Byzantine behavior
trustMgr.RecordInteraction(
    "node-1", "byzantine-node",
    ByzantineBehavior,
    "Conflicting messages detected"
)

// Get trust score
score := trustMgr.GetTrustScore("byzantine-node")
// Result: 0.0 (isolated)
```

---

## Monitoring & Metrics

### PBFT Metrics
```go
metrics := pbft.GetMetrics()
// Returns:
// - View (current view number)
// - CheckpointSeq (latest checkpoint)
// - ExecutedRequests (total executed)
// - ByzantineTolerance (f value)
```

### ProBFT Metrics
```go
state := probft.GetState()
// Returns:
// - Phase (current consensus phase)
// - Height (block height)
// - QuorumSize (required votes)
```

### TP-BFT Metrics
```go
metrics := tpbft.GetMetrics()
// Returns:
// - view, sequence
// - committee_size, f
// - throughput, consensus_latency
// - prepared_count, committed_count
```

### BullShark Metrics
```go
metrics := bullshark.GetMetrics()
// Returns:
// - Round (current round)
// - TxThroughput (tx/sec)
// - ProposalCount, CommitCount
```

---

## Common Issues & Solutions

### Issue: "Byzantine tolerance insufficient"
**Cause**: n < 3f+1
**Solution**: Add more nodes or reduce Byzantine assumption
```go
minNodes := CalculateMinimumNodes(f)  // Returns 3f+1
```

### Issue: "Quorum not reached"
**Cause**: Too few active nodes or network partition
**Solution**: Check network connectivity, adjust quorum threshold
```go
// Adaptive quorum for degraded conditions
condition := NetworkCondition{
    ActiveNodes:    currentActive,
    FailureRate:    measuredFailureRate,
}
result, _ := CalculateAdaptiveQuorum(config, condition)
```

### Issue: "VRF proof verification failed"
**Cause**: Wrong public key or tampered proof
**Solution**: Verify node identity and key distribution
```go
valid := VerifyVRF(expectedPubKey, input, proof)
if !valid {
    // Reject message, log Byzantine behavior
}
```

### Issue: "Trust score manipulation"
**Cause**: Byzantine nodes gaming reputation
**Solution**: EigenTrust algorithm resistant to gaming
```go
// Recompute global trust with pre-trusted peers
trustMgr.SetPreTrustedNode("trusted-bootstrap", 1.0)
trustMgr.RecomputeTrust()
```

---

## Development Checklist

### Implementing New Consensus Protocol

- [ ] Define message types (pre-prepare, prepare, commit, etc.)
- [ ] Implement Byzantine tolerance calculation (f < n/3)
- [ ] Add cryptographic verification (signatures, digests)
- [ ] Implement quorum logic (2f+1 or custom)
- [ ] Add view-change/leader rotation mechanism
- [ ] Implement checkpoint and garbage collection
- [ ] Add metrics collection
- [ ] Write comprehensive tests:
  - [ ] Byzantine tolerance limits
  - [ ] Message validation
  - [ ] Quorum intersection
  - [ ] Attack scenarios
  - [ ] Performance benchmarks

### Security Audit Checklist

- [ ] Message authentication implemented
- [ ] Replay attack prevention
- [ ] Sybil attack mitigation
- [ ] Fork attack prevention
- [ ] DoS/rate limiting
- [ ] Byzantine node isolation
- [ ] Quorum intersection validation
- [ ] Cryptographic key management
- [ ] Secure random number generation
- [ ] Timeout and liveness guarantees

---

## Further Reading

**Internal Documentation**:
- `/docs/dwcp/BYZANTINE_CONSENSUS_VALIDATION_REPORT.md` (Complete analysis)
- `/backend/core/network/dwcp/v3/consensus/README.md` (If exists)
- `/docs/architecture/ARCHITECTURE_SUMMARY.md`

**Academic Papers**:
- PBFT: "Practical Byzantine Fault Tolerance" (Castro & Liskov, 1999)
- ProBFT: Probabilistic Byzantine Fault Tolerance research
- EigenTrust: "The EigenTrust Algorithm for Reputation Management" (Kamvar et al., 2003)
- BullShark: "Bullshark: DAG BFT Protocols Made Practical" (Spiegelman et al., 2022)

**Standards**:
- Byzantine Fault Tolerance: 33% maximum (n ≥ 3f+1)
- Quorum Intersection: 2q - n ≥ f+1
- VRF: RFC 9381 (Verifiable Random Functions)

---

**Last Updated**: 2025-11-14
**Maintained By**: Byzantine Consensus Coordinator
**Status**: Production Ready (with noted improvements)
