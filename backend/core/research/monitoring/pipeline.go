package monitoring

import (
	"context"
	"encoding/xml"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ResearchPaper represents a research paper from various sources
type ResearchPaper struct {
	ID          string
	Title       string
	Authors     []string
	Abstract    string
	Categories  []string
	PublishDate time.Time
	Source      string // arxiv, conference, journal
	URL         string
	PDFLink     string

	// Analysis
	RelevanceScore float64
	Keywords       []string
	CitationCount  int
	Implemented    bool
}

// ArxivEntry represents an arXiv feed entry
type ArxivEntry struct {
	ID        string   `xml:"id"`
	Updated   string   `xml:"updated"`
	Published string   `xml:"published"`
	Title     string   `xml:"title"`
	Summary   string   `xml:"summary"`
	Authors   []Author `xml:"author"`
	Categories []Category `xml:"category"`
	Link      []Link   `xml:"link"`
}

type Author struct {
	Name string `xml:"name"`
}

type Category struct {
	Term string `xml:"term,attr"`
}

type Link struct {
	Href  string `xml:"href,attr"`
	Rel   string `xml:"rel,attr"`
	Type  string `xml:"type,attr"`
	Title string `xml:"title,attr"`
}

type ArxivFeed struct {
	XMLName xml.Name     `xml:"feed"`
	Entries []ArxivEntry `xml:"entry"`
}

// ResearchMonitor monitors research publications
type ResearchMonitor struct {
	config       MonitorConfig
	papers       map[string]*ResearchPaper
	researchers  map[string][]string // researcher -> papers
	mu           sync.RWMutex
	client       *http.Client
	subscribers  []chan *ResearchPaper
}

// MonitorConfig configures research monitoring
type MonitorConfig struct {
	ArxivCategories   []string
	Keywords          []string
	KeyResearchers    []string
	Conferences       []string
	MonitoringInterval time.Duration
	MaxPapersPerDay   int
	MinRelevanceScore float64
}

// NewResearchMonitor creates a new research monitor
func NewResearchMonitor(config MonitorConfig) *ResearchMonitor {
	return &ResearchMonitor{
		config:      config,
		papers:      make(map[string]*ResearchPaper),
		researchers: make(map[string][]string),
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		subscribers: make([]chan *ResearchPaper, 0),
	}
}

// Start begins monitoring research publications
func (rm *ResearchMonitor) Start(ctx context.Context) error {
	ticker := time.NewTicker(rm.config.MonitoringInterval)
	defer ticker.Stop()

	// Initial scan
	if err := rm.scanArxiv(ctx); err != nil {
		return fmt.Errorf("initial scan failed: %w", err)
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := rm.scanArxiv(ctx); err != nil {
				fmt.Printf("scan error: %v\n", err)
			}
		}
	}
}

// scanArxiv scans arXiv for new papers
func (rm *ResearchMonitor) scanArxiv(ctx context.Context) error {
	for _, category := range rm.config.ArxivCategories {
		if err := rm.scanArxivCategory(ctx, category); err != nil {
			return err
		}
	}
	return nil
}

// scanArxivCategory scans a specific arXiv category
func (rm *ResearchMonitor) scanArxivCategory(ctx context.Context, category string) error {
	// Build query
	query := fmt.Sprintf("cat:%s", category)

	// Add keyword filters
	if len(rm.config.Keywords) > 0 {
		keywords := strings.Join(rm.config.Keywords, "+OR+")
		query = fmt.Sprintf("%s+AND+(%s)", query, keywords)
	}

	url := fmt.Sprintf(
		"http://export.arxiv.org/api/query?search_query=%s&sortBy=submittedDate&sortOrder=descending&max_results=%d",
		query,
		rm.config.MaxPapersPerDay,
	)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	resp, err := rm.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var feed ArxivFeed
	if err := xml.NewDecoder(resp.Body).Decode(&feed); err != nil {
		return err
	}

	// Process entries
	for _, entry := range feed.Entries {
		paper := rm.convertArxivEntry(entry)

		// Calculate relevance
		paper.RelevanceScore = rm.calculateRelevance(paper)

		if paper.RelevanceScore >= rm.config.MinRelevanceScore {
			rm.addPaper(paper)
		}
	}

	return nil
}

// convertArxivEntry converts an arXiv entry to a research paper
func (rm *ResearchMonitor) convertArxivEntry(entry ArxivEntry) *ResearchPaper {
	paper := &ResearchPaper{
		ID:         entry.ID,
		Title:      strings.TrimSpace(entry.Title),
		Abstract:   strings.TrimSpace(entry.Summary),
		Categories: make([]string, len(entry.Categories)),
		Source:     "arxiv",
		URL:        entry.ID,
	}

	// Parse date
	if t, err := time.Parse(time.RFC3339, entry.Published); err == nil {
		paper.PublishDate = t
	}

	// Extract authors
	paper.Authors = make([]string, len(entry.Authors))
	for i, author := range entry.Authors {
		paper.Authors[i] = author.Name
	}

	// Extract categories
	for i, cat := range entry.Categories {
		paper.Categories[i] = cat.Term
	}

	// Find PDF link
	for _, link := range entry.Link {
		if link.Title == "pdf" {
			paper.PDFLink = link.Href
			break
		}
	}

	// Extract keywords
	paper.Keywords = rm.extractKeywords(paper.Title + " " + paper.Abstract)

	return paper
}

// calculateRelevance calculates relevance score for a paper
func (rm *ResearchMonitor) calculateRelevance(paper *ResearchPaper) float64 {
	score := 0.0

	// Keyword matching
	text := strings.ToLower(paper.Title + " " + paper.Abstract)
	for _, keyword := range rm.config.Keywords {
		if strings.Contains(text, strings.ToLower(keyword)) {
			score += 0.1
		}
	}

	// Key researcher matching
	for _, author := range paper.Authors {
		for _, researcher := range rm.config.KeyResearchers {
			if strings.Contains(strings.ToLower(author), strings.ToLower(researcher)) {
				score += 0.3
				break
			}
		}
	}

	// Category relevance
	for _, cat := range paper.Categories {
		for _, targetCat := range rm.config.ArxivCategories {
			if cat == targetCat {
				score += 0.05
				break
			}
		}
	}

	// Recency bonus
	daysSince := time.Since(paper.PublishDate).Hours() / 24
	if daysSince < 7 {
		score += 0.1
	}

	// Cap at 1.0
	if score > 1.0 {
		score = 1.0
	}

	return score
}

// extractKeywords extracts keywords from text
func (rm *ResearchMonitor) extractKeywords(text string) []string {
	keywords := make([]string, 0)
	text = strings.ToLower(text)

	keywordList := []string{
		"consensus", "distributed", "federated", "quantum",
		"cryptography", "machine learning", "neural network",
		"blockchain", "security", "privacy", "encryption",
		"edge computing", "serverless", "microservices",
	}

	for _, keyword := range keywordList {
		if strings.Contains(text, keyword) {
			keywords = append(keywords, keyword)
		}
	}

	return keywords
}

// addPaper adds a paper to the monitor
func (rm *ResearchMonitor) addPaper(paper *ResearchPaper) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	if _, exists := rm.papers[paper.ID]; exists {
		return
	}

	rm.papers[paper.ID] = paper

	// Index by researcher
	for _, author := range paper.Authors {
		rm.researchers[author] = append(rm.researchers[author], paper.ID)
	}

	// Notify subscribers
	for _, ch := range rm.subscribers {
		select {
		case ch <- paper:
		default:
		}
	}
}

// Subscribe subscribes to new papers
func (rm *ResearchMonitor) Subscribe() <-chan *ResearchPaper {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	ch := make(chan *ResearchPaper, 100)
	rm.subscribers = append(rm.subscribers, ch)
	return ch
}

// GetPapers returns papers matching criteria
func (rm *ResearchMonitor) GetPapers(filter PaperFilter) []*ResearchPaper {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	papers := make([]*ResearchPaper, 0)
	for _, paper := range rm.papers {
		if filter.Matches(paper) {
			papers = append(papers, paper)
		}
	}

	return papers
}

// PaperFilter filters research papers
type PaperFilter struct {
	MinRelevance float64
	Categories   []string
	Authors      []string
	Keywords     []string
	Since        time.Time
}

// Matches checks if a paper matches the filter
func (f PaperFilter) Matches(paper *ResearchPaper) bool {
	if paper.RelevanceScore < f.MinRelevance {
		return false
	}

	if !f.Since.IsZero() && paper.PublishDate.Before(f.Since) {
		return false
	}

	if len(f.Categories) > 0 {
		found := false
		for _, cat := range paper.Categories {
			for _, filterCat := range f.Categories {
				if cat == filterCat {
					found = true
					break
				}
			}
		}
		if !found {
			return false
		}
	}

	if len(f.Authors) > 0 {
		found := false
		for _, author := range paper.Authors {
			for _, filterAuthor := range f.Authors {
				if strings.Contains(strings.ToLower(author), strings.ToLower(filterAuthor)) {
					found = true
					break
				}
			}
		}
		if !found {
			return false
		}
	}

	if len(f.Keywords) > 0 {
		found := false
		for _, keyword := range paper.Keywords {
			for _, filterKeyword := range f.Keywords {
				if keyword == filterKeyword {
					found = true
					break
				}
			}
		}
		if !found {
			return false
		}
	}

	return true
}

// GetStats returns monitoring statistics
func (rm *ResearchMonitor) GetStats() MonitorStats {
	rm.mu.RLock()
	defer rm.mu.RUnlock()

	stats := MonitorStats{
		TotalPapers: len(rm.papers),
		PapersByCategory: make(map[string]int),
		PapersByAuthor: make(map[string]int),
	}

	for _, paper := range rm.papers {
		for _, cat := range paper.Categories {
			stats.PapersByCategory[cat]++
		}
		for _, author := range paper.Authors {
			stats.PapersByAuthor[author]++
		}
		if paper.Implemented {
			stats.Implemented++
		}
	}

	return stats
}

// MonitorStats contains monitoring statistics
type MonitorStats struct {
	TotalPapers       int
	Implemented       int
	PapersByCategory  map[string]int
	PapersByAuthor    map[string]int
}
