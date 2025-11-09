package monitoring

import (
	"context"
	"testing"
	"time"
)

func TestNewResearchMonitor(t *testing.T) {
	config := MonitorConfig{
		ArxivCategories:   []string{"cs.DC", "cs.NI"},
		Keywords:          []string{"distributed", "consensus"},
		MonitoringInterval: time.Hour,
		MaxPapersPerDay:   50,
		MinRelevanceScore: 0.5,
	}

	monitor := NewResearchMonitor(config)

	if monitor == nil {
		t.Fatal("NewResearchMonitor returned nil")
	}

	if len(monitor.papers) != 0 {
		t.Errorf("Expected 0 papers, got %d", len(monitor.papers))
	}
}

func TestAddPaper(t *testing.T) {
	config := MonitorConfig{
		ArxivCategories:   []string{"cs.DC"},
		MinRelevanceScore: 0.5,
	}

	monitor := NewResearchMonitor(config)

	paper := &ResearchPaper{
		ID:             "test-1",
		Title:          "Test Paper",
		Authors:        []string{"Author 1"},
		Abstract:       "Test abstract",
		Categories:     []string{"cs.DC"},
		PublishDate:    time.Now(),
		RelevanceScore: 0.8,
	}

	monitor.addPaper(paper)

	if len(monitor.papers) != 1 {
		t.Errorf("Expected 1 paper, got %d", len(monitor.papers))
	}

	retrieved := monitor.papers["test-1"]
	if retrieved.Title != "Test Paper" {
		t.Errorf("Expected title 'Test Paper', got %s", retrieved.Title)
	}
}

func TestCalculateRelevance(t *testing.T) {
	config := MonitorConfig{
		Keywords: []string{"distributed", "consensus"},
	}

	monitor := NewResearchMonitor(config)

	paper := &ResearchPaper{
		Title:    "Distributed Consensus Algorithms",
		Abstract: "A study of distributed consensus protocols",
		Authors:  []string{"John Doe"},
		Categories: []string{"cs.DC"},
		PublishDate: time.Now(),
	}

	score := monitor.calculateRelevance(paper)

	if score <= 0 {
		t.Error("Expected relevance score > 0")
	}

	if score > 1.0 {
		t.Error("Expected relevance score <= 1.0")
	}
}

func TestPaperFilter(t *testing.T) {
	filter := PaperFilter{
		MinRelevance: 0.5,
		Categories:   []string{"cs.DC"},
		Since:        time.Now().Add(-7 * 24 * time.Hour),
	}

	paper1 := &ResearchPaper{
		RelevanceScore: 0.8,
		Categories:     []string{"cs.DC"},
		PublishDate:    time.Now().Add(-1 * 24 * time.Hour),
	}

	paper2 := &ResearchPaper{
		RelevanceScore: 0.3,
		Categories:     []string{"cs.DC"},
		PublishDate:    time.Now(),
	}

	if !filter.Matches(paper1) {
		t.Error("Expected paper1 to match filter")
	}

	if filter.Matches(paper2) {
		t.Error("Expected paper2 not to match filter")
	}
}

func TestSubscribe(t *testing.T) {
	config := MonitorConfig{
		MinRelevanceScore: 0.5,
	}

	monitor := NewResearchMonitor(config)
	ch := monitor.Subscribe()

	if ch == nil {
		t.Fatal("Subscribe returned nil channel")
	}

	// Add paper asynchronously
	go func() {
		time.Sleep(100 * time.Millisecond)
		paper := &ResearchPaper{
			ID:             "test-2",
			Title:          "Test Paper 2",
			RelevanceScore: 0.8,
		}
		monitor.addPaper(paper)
	}()

	// Wait for notification
	select {
	case paper := <-ch:
		if paper.ID != "test-2" {
			t.Errorf("Expected paper ID 'test-2', got %s", paper.ID)
		}
	case <-time.After(500 * time.Millisecond):
		t.Error("Timeout waiting for paper notification")
	}
}

func TestGetStats(t *testing.T) {
	config := MonitorConfig{}
	monitor := NewResearchMonitor(config)

	// Add test papers
	papers := []*ResearchPaper{
		{
			ID:         "p1",
			Categories: []string{"cs.DC"},
			Authors:    []string{"Author 1"},
			Implemented: true,
		},
		{
			ID:         "p2",
			Categories: []string{"cs.NI"},
			Authors:    []string{"Author 1", "Author 2"},
			Implemented: false,
		},
	}

	for _, paper := range papers {
		monitor.addPaper(paper)
	}

	stats := monitor.GetStats()

	if stats.TotalPapers != 2 {
		t.Errorf("Expected 2 total papers, got %d", stats.TotalPapers)
	}

	if stats.Implemented != 1 {
		t.Errorf("Expected 1 implemented paper, got %d", stats.Implemented)
	}

	if stats.PapersByCategory["cs.DC"] != 1 {
		t.Errorf("Expected 1 cs.DC paper, got %d", stats.PapersByCategory["cs.DC"])
	}
}

func TestConcurrentAccess(t *testing.T) {
	config := MonitorConfig{}
	monitor := NewResearchMonitor(config)

	// Concurrent writes
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(id int) {
			paper := &ResearchPaper{
				ID:             string(rune('A' + id)),
				Title:          "Concurrent Paper",
				RelevanceScore: 0.5,
			}
			monitor.addPaper(paper)
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// Concurrent reads
	for i := 0; i < 10; i++ {
		go func() {
			_ = monitor.GetStats()
			done <- true
		}()
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestStartMonitoring(t *testing.T) {
	config := MonitorConfig{
		ArxivCategories:   []string{"cs.DC"},
		MonitoringInterval: 100 * time.Millisecond,
		MaxPapersPerDay:   10,
		MinRelevanceScore: 0.5,
	}

	monitor := NewResearchMonitor(config)
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	// Start monitoring in background
	errCh := make(chan error, 1)
	go func() {
		errCh <- monitor.Start(ctx)
	}()

	// Wait for completion or timeout
	select {
	case err := <-errCh:
		if err != context.DeadlineExceeded && err != context.Canceled {
			t.Errorf("Unexpected error: %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Error("Monitor did not stop in time")
	}
}
