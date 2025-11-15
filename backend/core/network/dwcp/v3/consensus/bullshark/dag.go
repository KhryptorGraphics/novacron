package bullshark

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// DAG represents a Directed Acyclic Graph structure for Bullshark consensus
type DAG struct {
	vertices map[string]*Vertex
	edges    map[string][]string
	roots    []*Vertex
	mu       sync.RWMutex

	// Performance metrics
	vertexCount   int64
	edgeCount     int64
	maxDepth      int
	committedTxs  int64
}

// Vertex represents a node in the DAG containing transactions
type Vertex struct {
	ID        string
	Round     int
	Txs       []Transaction
	Parents   []*Vertex
	ParentIDs []string
	Timestamp time.Time
	Author    string

	// Consensus state
	Committed bool
	Depth     int
	Weight    int64

	mu sync.RWMutex
}

// Transaction represents a single transaction in the DAG
type Transaction struct {
	ID        string
	Data      []byte
	Sender    string
	Recipient string
	Amount    int64
	Nonce     int64
	Timestamp time.Time
}

// NewDAG creates a new Directed Acyclic Graph
func NewDAG() *DAG {
	return &DAG{
		vertices: make(map[string]*Vertex),
		edges:    make(map[string][]string),
		roots:    make([]*Vertex, 0),
		maxDepth: 0,
	}
}

// AddVertex adds a new vertex to the DAG with validation
func (d *DAG) AddVertex(v *Vertex) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Validate vertex
	if v == nil {
		return fmt.Errorf("vertex cannot be nil")
	}

	if v.ID == "" {
		return fmt.Errorf("vertex ID cannot be empty")
	}

	// Check for duplicate
	if _, exists := d.vertices[v.ID]; exists {
		return fmt.Errorf("vertex %s already exists", v.ID)
	}

	// Validate parents exist (except for genesis)
	if v.Round > 0 {
		for _, parentID := range v.ParentIDs {
			if _, exists := d.vertices[parentID]; !exists {
				return fmt.Errorf("parent vertex %s does not exist", parentID)
			}
		}
	}

	// Detect cycles (basic check)
	if d.wouldCreateCycle(v) {
		return fmt.Errorf("adding vertex would create a cycle")
	}

	// Add vertex
	d.vertices[v.ID] = v
	d.vertexCount++

	// Add edges
	for _, parentID := range v.ParentIDs {
		d.edges[parentID] = append(d.edges[parentID], v.ID)
		d.edgeCount++
	}

	// Update roots
	if v.Round == 0 || len(v.Parents) == 0 {
		d.roots = append(d.roots, v)
	}

	// Update depth
	if v.Depth > d.maxDepth {
		d.maxDepth = v.Depth
	}

	return nil
}

// GetVertex retrieves a vertex by ID
func (d *DAG) GetVertex(id string) (*Vertex, bool) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	v, exists := d.vertices[id]
	return v, exists
}

// GetChildren returns all children of a vertex
func (d *DAG) GetChildren(id string) []*Vertex {
	d.mu.RLock()
	defer d.mu.RUnlock()

	childIDs, exists := d.edges[id]
	if !exists {
		return nil
	}

	children := make([]*Vertex, 0, len(childIDs))
	for _, childID := range childIDs {
		if child, ok := d.vertices[childID]; ok {
			children = append(children, child)
		}
	}

	return children
}

// GetRoots returns all root vertices
func (d *DAG) GetRoots() []*Vertex {
	d.mu.RLock()
	defer d.mu.RUnlock()

	roots := make([]*Vertex, len(d.roots))
	copy(roots, d.roots)
	return roots
}

// GetVerticesByRound returns all vertices in a specific round
func (d *DAG) GetVerticesByRound(round int) []*Vertex {
	d.mu.RLock()
	defer d.mu.RUnlock()

	vertices := make([]*Vertex, 0)
	for _, v := range d.vertices {
		if v.Round == round {
			vertices = append(vertices, v)
		}
	}

	return vertices
}

// wouldCreateCycle checks if adding a vertex would create a cycle
func (d *DAG) wouldCreateCycle(v *Vertex) bool {
	visited := make(map[string]bool)

	var hasCycle func(string) bool
	hasCycle = func(nodeID string) bool {
		if nodeID == v.ID {
			return true
		}

		if visited[nodeID] {
			return false
		}

		visited[nodeID] = true

		for _, childID := range d.edges[nodeID] {
			if hasCycle(childID) {
				return true
			}
		}

		return false
	}

	for _, parentID := range v.ParentIDs {
		if hasCycle(parentID) {
			return true
		}
	}

	return false
}

// CalculateWeight calculates the weight of a vertex based on descendants
func (d *DAG) CalculateWeight(v *Vertex) int64 {
	d.mu.RLock()
	defer d.mu.RUnlock()

	visited := make(map[string]bool)
	var weight int64 = 1 // Count self

	var traverse func(string)
	traverse = func(nodeID string) {
		if visited[nodeID] {
			return
		}
		visited[nodeID] = true
		weight++

		for _, childID := range d.edges[nodeID] {
			traverse(childID)
		}
	}

	for _, childID := range d.edges[v.ID] {
		traverse(childID)
	}

	return weight
}

// TopologicalSort returns vertices in topological order
func (d *DAG) TopologicalSort() ([]*Vertex, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	inDegree := make(map[string]int)
	for id := range d.vertices {
		inDegree[id] = 0
	}

	for _, childIDs := range d.edges {
		for _, childID := range childIDs {
			inDegree[childID]++
		}
	}

	queue := make([]*Vertex, 0)
	for id, degree := range inDegree {
		if degree == 0 {
			if v, ok := d.vertices[id]; ok {
				queue = append(queue, v)
			}
		}
	}

	sorted := make([]*Vertex, 0, len(d.vertices))

	for len(queue) > 0 {
		v := queue[0]
		queue = queue[1:]
		sorted = append(sorted, v)

		for _, childID := range d.edges[v.ID] {
			inDegree[childID]--
			if inDegree[childID] == 0 {
				if child, ok := d.vertices[childID]; ok {
					queue = append(queue, child)
				}
			}
		}
	}

	if len(sorted) != len(d.vertices) {
		return nil, fmt.Errorf("cycle detected in DAG")
	}

	return sorted, nil
}

// Metrics returns DAG performance metrics
func (d *DAG) Metrics() map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return map[string]interface{}{
		"vertex_count":   d.vertexCount,
		"edge_count":     d.edgeCount,
		"max_depth":      d.maxDepth,
		"committed_txs":  d.committedTxs,
		"root_count":     len(d.roots),
	}
}

// NewVertex creates a new vertex with the given parameters
func NewVertex(author string, round int, txs []Transaction, parents []*Vertex) *Vertex {
	v := &Vertex{
		ID:        generateVertexID(author, round, txs),
		Round:     round,
		Txs:       txs,
		Parents:   parents,
		ParentIDs: make([]string, len(parents)),
		Timestamp: time.Now(),
		Author:    author,
		Committed: false,
		Depth:     0,
		Weight:    1,
	}

	// Extract parent IDs
	for i, p := range parents {
		v.ParentIDs[i] = p.ID
	}

	// Calculate depth
	maxParentDepth := 0
	for _, p := range parents {
		if p.Depth > maxParentDepth {
			maxParentDepth = p.Depth
		}
	}
	v.Depth = maxParentDepth + 1

	return v
}

// generateVertexID creates a unique ID for a vertex
func generateVertexID(author string, round int, txs []Transaction) string {
	h := sha256.New()
	h.Write([]byte(author))
	h.Write([]byte(fmt.Sprintf("%d", round)))
	h.Write([]byte(fmt.Sprintf("%d", time.Now().UnixNano())))

	for _, tx := range txs {
		h.Write([]byte(tx.ID))
	}

	return hex.EncodeToString(h.Sum(nil))[:16]
}

// NewTransaction creates a new transaction
func NewTransaction(sender, recipient string, amount int64, data []byte) *Transaction {
	return &Transaction{
		ID:        generateTxID(sender, recipient, amount),
		Data:      data,
		Sender:    sender,
		Recipient: recipient,
		Amount:    amount,
		Nonce:     time.Now().UnixNano(),
		Timestamp: time.Now(),
	}
}

// generateTxID creates a unique ID for a transaction
func generateTxID(sender, recipient string, amount int64) string {
	h := sha256.New()
	h.Write([]byte(sender))
	h.Write([]byte(recipient))
	h.Write([]byte(fmt.Sprintf("%d", amount)))
	h.Write([]byte(fmt.Sprintf("%d", time.Now().UnixNano())))

	return hex.EncodeToString(h.Sum(nil))[:16]
}
