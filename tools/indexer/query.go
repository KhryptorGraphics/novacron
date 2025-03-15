package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/qdrant/go-client/qdrant"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	SearchLimit       = 10  // Max number of results to return
	MinScoreThreshold = 0.6 // Minimum score to consider a result relevant
)

// Result represents a search result
type Result struct {
	Path    string
	Score   float32
	Content string
	Excerpt string
}

func queryCommand() {
	// Define flags
	var query string
	var filter string
	var ext string
	var showContent bool

	flag.StringVar(&query, "q", "", "Search query (required)")
	flag.StringVar(&filter, "path", "", "Filter results by path prefix (optional)")
	flag.StringVar(&ext, "ext", "", "Filter results by file extension (optional)")
	flag.BoolVar(&showContent, "content", false, "Show full file content in results")
	flag.Parse()

	// Require query parameter
	if query == "" {
		fmt.Println("Error: query parameter (-q) is required")
		fmt.Println("Usage: go run query.go -q \"your search query\"")
		os.Exit(1)
	}

	// Connect to Qdrant
	ctx := context.Background()
	conn, err := grpc.Dial(
		fmt.Sprintf("%s:%d", "localhost", 6333),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		fmt.Printf("Failed to connect to Qdrant: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	// Create Qdrant points client
	pointsClient := qdrant.NewPointsClient(conn)

	// Initialize OpenAI client
	openaiClient := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	if os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Println("Warning: OPENAI_API_KEY environment variable not set, using mock embeddings")
	}

	// Generate embedding for query
	queryEmbedding, err := getEmbeddings(ctx, openaiClient, query)
	if err != nil {
		fmt.Printf("Error generating embeddings: %v\n", err)
		os.Exit(1)
	}

	// Build filters
	var filter_conditions []*qdrant.Condition

	if filter != "" {
		filter_conditions = append(filter_conditions, &qdrant.Condition{
			ConditionOneOf: &qdrant.Condition_Field{
				Field: &qdrant.FieldCondition{
					Key: "path",
					Match: &qdrant.Match{
						MatchValue: &qdrant.Match_Text{
							Text: &qdrant.MatchText{
								Text: filter,
							},
						},
					},
				},
			},
		})
	}

	if ext != "" {
		// Ensure extension starts with a dot
		if !strings.HasPrefix(ext, ".") {
			ext = "." + ext
		}

		filter_conditions = append(filter_conditions, &qdrant.Condition{
			ConditionOneOf: &qdrant.Condition_Field{
				Field: &qdrant.FieldCondition{
					Key: "extension",
					Match: &qdrant.Match{
						MatchValue: &qdrant.Match_Keyword{
							Keyword: &qdrant.MatchKeyword{
								Keyword: ext,
							},
						},
					},
				},
			},
		})
	}

	var filter_params *qdrant.Filter
	if len(filter_conditions) > 0 {
		filter_params = &qdrant.Filter{
			Must: filter_conditions,
		}
	}

	// Prepare a map for the vector search
	vectorMap := make(map[string]*qdrant.Vector)
	vectorMap[""] = &qdrant.Vector{Data: queryEmbedding}

	vectorName := ""
	threshold := float32(MinScoreThreshold)

	// Search Qdrant
	searchParams := qdrant.SearchPoints{
		CollectionName: CollectionName,
		Vector:         vectorMap[""],
		Limit:          uint64(SearchLimit),
		WithPayload: &qdrant.WithPayloadSelector{
			SelectorOptions: &qdrant.WithPayloadSelector_Include{
				Include: &qdrant.PayloadIncludeSelector{
					Fields: []string{"path", "content"},
				},
			},
		},
		Filter:         filter_params,
		ScoreThreshold: &threshold,
		VectorName:     &vectorName,
	}

	searchResults, err := pointsClient.Search(ctx, &searchParams)
	if err != nil {
		fmt.Printf("Error searching Qdrant: %v\n", err)
		os.Exit(1)
	}

	// Process results
	var results []Result
	for _, point := range searchResults.GetResult() {
		path := point.GetPayload()["path"].GetKind().(*qdrant.Value_StringValue).StringValue
		content := point.GetPayload()["content"].GetKind().(*qdrant.Value_StringValue).StringValue

		// Create excerpt (context around matches)
		excerpt := createExcerpt(content, query, 150)

		results = append(results, Result{
			Path:    path,
			Score:   point.GetScore(),
			Content: content,
			Excerpt: excerpt,
		})
	}

	// Sort results by score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Display results
	fmt.Printf("Found %d results for query: %s\n\n", len(results), query)
	for i, result := range results {
		fmt.Printf("%d. %s (Score: %.4f)\n", i+1, result.Path, result.Score)
		fmt.Printf("   %s\n\n", result.Excerpt)

		if showContent {
			fmt.Printf("--- Full Content ---\n")
			fmt.Printf("%s\n", result.Content)
			fmt.Printf("-------------------\n\n")
		}
	}
}

// createExcerpt generates a relevant excerpt from the content
func createExcerpt(content, query string, maxLength int) string {
	lowerContent := strings.ToLower(content)
	lowerQuery := strings.ToLower(query)

	// Find position of query terms
	var bestPos int
	bestScore := -1
	queryWords := strings.Fields(lowerQuery)

	// Find best matching location
	for pos := 0; pos < len(lowerContent); pos++ {
		score := 0
		for _, word := range queryWords {
			if pos+len(word) <= len(lowerContent) && strings.Contains(lowerContent[pos:pos+len(word)*2], word) {
				score++
			}
		}
		if score > bestScore {
			bestScore = score
			bestPos = pos
		}
	}

	// If no match, use beginning
	if bestScore == 0 {
		bestPos = 0
	}

	// Find closest line break before position
	start := bestPos
	for start > 0 && content[start] != '\n' {
		start--
	}
	if start > 0 {
		start++
	}

	// Find a good endpoint (try to end at line break or punctuation)
	end := start + maxLength
	if end > len(content) {
		end = len(content)
	} else {
		// Try to end at a line break
		for i := end; i < len(content) && i < end+50; i++ {
			if content[i] == '\n' {
				end = i
				break
			}
		}
	}

	excerpt := content[start:end]

	// Add ellipsis if truncated
	if start > 0 {
		excerpt = "..." + excerpt
	}
	if end < len(content) {
		excerpt = excerpt + "..."
	}

	return excerpt
}

// If run directly, use the query command
func init() {
	// Check if this file is being directly executed versus imported
	if len(os.Args) > 0 && strings.Contains(os.Args[0], "query") {
		queryCommand()
		os.Exit(0)
	}
}
