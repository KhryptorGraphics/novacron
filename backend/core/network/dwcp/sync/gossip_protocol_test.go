package sync

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/sync/crdt"
	"go.uber.org/zap"
)

type gossipTestTransport struct{}

func (g *gossipTestTransport) Send(peer *RegionPeer, message *Message) error { return nil }
func (g *gossipTestTransport) Receive() (*Message, error)                    { return nil, nil }
func (g *gossipTestTransport) Close() error                                  { return nil }

func TestGossipApplyUpdateCreatesMissingCRDT(t *testing.T) {
	engine := NewASSEngine("node-a", &gossipTestTransport{}, zap.NewNop())
	counter := crdt.NewGCounter("node-b")
	counter.Increment(7)
	data, err := counter.Marshal()
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	update := &CRDTUpdate{
		Key:       "counter",
		Type:      "g_counter",
		Data:      json.RawMessage(data),
		Timestamp: time.Now(),
	}

	if err := engine.gossip.applyUpdate(update); err != nil {
		t.Fatalf("applyUpdate failed: %v", err)
	}
	value, ok := engine.Get("counter")
	if !ok {
		t.Fatal("counter was not stored")
	}
	if got := value.Value(); got != uint64(7) {
		t.Fatalf("counter value = %v, want 7", got)
	}
}

func TestGossipCreateCRDTRejectsInvalidPayload(t *testing.T) {
	engine := NewASSEngine("node-a", &gossipTestTransport{}, zap.NewNop())
	if value := engine.gossip.createCRDT("g_counter", json.RawMessage(`{"bad"`)); value != nil {
		t.Fatal("expected invalid CRDT payload to be rejected")
	}
	if value := engine.gossip.createCRDT("unknown", json.RawMessage(`{}`)); value != nil {
		t.Fatal("expected unknown CRDT type to be rejected")
	}
}
