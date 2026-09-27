package audit

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// Concurrent writers and readers must not race (run with -race); the
// canonical api-server shares one SimpleAuditLogger across HTTP handlers and
// the security SSE stream.
func TestSimpleAuditLoggerConcurrentLogAndQuery(t *testing.T) {
	logger := NewSimpleAuditLogger()
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				if err := logger.LogEvent(ctx, &AuditEvent{Actor: fmt.Sprintf("actor-%d", i)}); err != nil {
					t.Errorf("LogEvent: %v", err)
				}
			}
		}(i)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				if _, err := logger.QueryEvents(ctx, &AuditFilter{}); err != nil {
					t.Errorf("QueryEvents: %v", err)
				}
				if _, err := logger.Query(ctx, AuditFilter{}); err != nil {
					t.Errorf("Query: %v", err)
				}
			}
		}()
	}
	wg.Wait()

	events, err := logger.QueryEvents(ctx, nil)
	if err != nil {
		t.Fatalf("QueryEvents: %v", err)
	}
	if len(events) != 800 {
		t.Fatalf("expected 800 events, got %d", len(events))
	}
}

func TestSimpleAuditLoggerDiscardsOldestBeyondCap(t *testing.T) {
	logger := NewSimpleAuditLogger()
	ctx := context.Background()
	total := simpleAuditLoggerMaxEvents + 5
	for i := 0; i < total; i++ {
		if err := logger.LogEvent(ctx, &AuditEvent{ID: fmt.Sprintf("evt-%d", i)}); err != nil {
			t.Fatalf("LogEvent: %v", err)
		}
	}

	events, err := logger.QueryEvents(ctx, nil)
	if err != nil {
		t.Fatalf("QueryEvents: %v", err)
	}
	if len(events) != simpleAuditLoggerMaxEvents {
		t.Fatalf("expected %d retained events, got %d", simpleAuditLoggerMaxEvents, len(events))
	}
	if events[0].ID != "evt-5" || events[len(events)-1].ID != fmt.Sprintf("evt-%d", total-1) {
		t.Fatalf("expected oldest events discarded, got first=%s last=%s", events[0].ID, events[len(events)-1].ID)
	}
}
