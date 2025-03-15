package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	QdrantHost      = "localhost"
	QdrantGRPCPort  = 6333 // Qdrant GRPC port
	CollectionName  = "novacron_files"
	VectorDimension = 1536 // OpenAI ada-002 dimension
	OpenAIMaxTokens = 8000 // Limit tokens to avoid exceeding API limits
	OpenAIModelName = "text-embedding-ada-002"
)

var (
	// Directories to skip during indexing
	skipDirs = map[string]bool{
		".git":         true,
		"node_modules": true,
		"vendor":       true,
		"dist":         true,
		"build":        true,
	}

	// File extensions to index
	indexExtensions = map[string]bool{
		".go":    true,
		".md":    true,
		".yml":   true,
		".yaml":  true,
		".json":  true,
		".proto": true,
		".sh":    true,
		".ps1":   true,
	}
)

// FileInfo holds metadata and content of a file
type FileInfo struct {
	Path         string    `json:"path"`
	Content      string    `json:"content"`
	Size         int64     `json:"size"`
	LastModified time.Time `json:"last_modified"`
	Hash         string    `json:"hash"`
	Extension    string    `json:"extension"`
}

func main() {
	// Initialize OpenAI client
	openaiClient := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	if os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Println("Warning: OPENAI_API_KEY environment variable not set, using mock embeddings")
	}

	// Connect to Qdrant
	ctx := context.Background()
	conn, err := grpc.Dial(
		fmt.Sprintf("%s:%d", QdrantHost, QdrantGRPCPort),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		fmt.Printf("Failed to connect to Qdrant: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()

	// Create Qdrant client
	collectionsClient := qdrant.NewCollectionsClient(conn)
	pointsClient := qdrant.NewPointsClient(conn)

	// Create collection if it doesn't exist
	createCollection(ctx, collectionsClient)

	// Start indexing from current directory
	startDir := "."
	if len(os.Args) > 1 {
		startDir = os.Args[1]
	}

	// Walk filesystem and index files
	indexFiles(ctx, openaiClient, pointsClient, startDir)

	fmt.Println("Indexing complete!")
}

// createCollection creates a new collection in Qdrant if it doesn't exist
func createCollection(ctx context.Context, client qdrant.CollectionsClient) {
	// Get list of collections to check if our collection exists
	listResp, err := client.List(ctx, &qdrant.ListCollectionsRequest{})
	if err != nil {
		fmt.Printf("Failed to list collections: %v\n", err)
		os.Exit(1)
	}

	// Check if collection exists
	collectionExists := false
	for _, collection := range listResp.GetCollections() {
		if collection.GetName() == CollectionName {
			collectionExists = true
			break
		}
	}

	if collectionExists {
		fmt.Printf("Collection %s already exists\n", CollectionName)
		return
	}

	// Create vector params
	distance := qdrant.Distance_Cosine
	size := uint64(VectorDimension)

	// Create collection
	_, err = client.Create(ctx, &qdrant.CreateCollection{
		CollectionName: CollectionName,
		VectorsConfig: &qdrant.VectorsConfig{
			Config: &qdrant.VectorsConfig_Params{
				Params: &qdrant.VectorParams{
					Size:     size,
					Distance: distance,
				},
			},
		},
	})
	if err != nil {
		fmt.Printf("Failed to create collection: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Collection %s created successfully\n", CollectionName)
}

// indexFiles walks the filesystem and indexes files
func indexFiles(ctx context.Context, openaiClient *openai.Client, pointsClient qdrant.PointsClient, rootDir string) {
	count := 0
	err := filepath.Walk(rootDir, func(path string, info fs.FileInfo, err error) error {
		if err != nil {
			fmt.Printf("Error accessing path %s: %v\n", path, err)
			return filepath.SkipDir
		}

		// Skip directories we want to ignore
		if info.IsDir() {
			if skipDirs[info.Name()] {
				fmt.Printf("Skipping directory: %s\n", path)
				return filepath.SkipDir
			}
			return nil
		}

		// Check if file extension should be indexed
		ext := filepath.Ext(info.Name())
		if !indexExtensions[ext] {
			return nil
		}

		// Clean path for display
		cleanPath := filepath.ToSlash(path)
		if strings.HasPrefix(cleanPath, "./") {
			cleanPath = cleanPath[2:]
		}

		// Read file content
		content, err := os.ReadFile(path)
		if err != nil {
			fmt.Printf("Error reading file %s: %v\n", path, err)
			return nil
		}

		// Skip empty files
		if len(content) == 0 {
			fmt.Printf("Skipping empty file: %s\n", cleanPath)
			return nil
		}

		// Create file hash
		hash := sha256.Sum256(content)
		hashString := hex.EncodeToString(hash[:])

		// Create FileInfo
		fileInfo := FileInfo{
			Path:         cleanPath,
			Content:      string(content),
			Size:         info.Size(),
			LastModified: info.ModTime(),
			Hash:         hashString,
			Extension:    ext,
		}

		// Generate embeddings
		embeddingValues, err := getEmbeddings(ctx, openaiClient, fileInfo.Content)
		if err != nil {
			fmt.Printf("Error generating embeddings for %s: %v\n", cleanPath, err)
			return nil
		}

		// Create point ID from file hash
		pointID, err := hashToUint64(fileInfo.Hash)
		if err != nil {
			fmt.Printf("Error creating point ID: %v\n", err)
			return nil
		}

		// Prepare payload with proper field names for Qdrant
		payload := make(map[string]*qdrant.Value)
		payload["path"] = &qdrant.Value{Kind: &qdrant.Value_StringValue{StringValue: fileInfo.Path}}
		payload["content"] = &qdrant.Value{Kind: &qdrant.Value_StringValue{StringValue: fileInfo.Content}}
		payload["size"] = &qdrant.Value{Kind: &qdrant.Value_IntegerValue{IntegerValue: fileInfo.Size}}
		payload["lastModified"] = &qdrant.Value{Kind: &qdrant.Value_IntegerValue{IntegerValue: fileInfo.LastModified.Unix()}}
		payload["hash"] = &qdrant.Value{Kind: &qdrant.Value_StringValue{StringValue: fileInfo.Hash}}
		payload["extension"] = &qdrant.Value{Kind: &qdrant.Value_StringValue{StringValue: fileInfo.Extension}}

		// Upsert point into Qdrant
		// Create vectors struct
		vectors := &qdrant.Vectors{
			VectorsOptions: &qdrant.Vectors_Vector{
				Vector: &qdrant.Vector{
					Data: embeddingValues,
				},
			},
		}

		_, err = pointsClient.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: CollectionName,
			Points: []*qdrant.PointStruct{
				{
					Id: &qdrant.PointId{
						PointIdOptions: &qdrant.PointId_Num{
							Num: pointID,
						},
					},
					Vectors: vectors,
					Payload: payload,
				},
			},
		})

		if err != nil {
			fmt.Printf("Error upserting point for %s: %v\n", cleanPath, err)
			return nil
		}

		count++
		fmt.Printf("Indexed file %d: %s\n", count, cleanPath)
		return nil
	})

	if err != nil {
		fmt.Printf("Error walking directory: %v\n", err)
	}
}

// getEmbeddings generates embeddings for text using OpenAI API
func getEmbeddings(ctx context.Context, client *openai.Client, text string) ([]float32, error) {
	// Truncate text if needed to avoid token limits
	if len(text) > OpenAIMaxTokens*4 {
		text = text[:OpenAIMaxTokens*4]
	}

	// Mock embeddings if no API key
	if os.Getenv("OPENAI_API_KEY") == "" {
		// Generate deterministic mock embedding based on text hash
		hash := sha256.Sum256([]byte(text))
		mockEmbedding := make([]float32, VectorDimension)
		for i := 0; i < VectorDimension && i < len(hash)*8; i++ {
			byteIndex := i / 8
			bitIndex := i % 8
			bit := (hash[byteIndex] >> bitIndex) & 1
			mockEmbedding[i] = float32(bit)
		}
		// Normalize mock embedding
		magnitude := float32(0)
		for _, v := range mockEmbedding {
			magnitude += v * v
		}
		magnitude = float32(1.0 / float64(magnitude))
		for i := range mockEmbedding {
			mockEmbedding[i] *= magnitude
		}
		return mockEmbedding, nil
	}

	// Get embeddings from OpenAI
	resp, err := client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{text},
		Model: openai.AdaEmbeddingV2, // Use the correct typed constant instead of string
	})

	if err != nil {
		return nil, fmt.Errorf("OpenAI API error: %v", err)
	}

	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}

	// Convert []float64 to []float32 for Qdrant
	embedding := make([]float32, len(resp.Data[0].Embedding))
	for i, v := range resp.Data[0].Embedding {
		embedding[i] = float32(v)
	}

	return embedding, nil
}

// hashToUint64 converts a hex string hash to a uint64 for point ID
func hashToUint64(hash string) (uint64, error) {
	if len(hash) < 16 {
		return 0, fmt.Errorf("hash too short: %s", hash)
	}
	// Use first 16 characters of hash (64 bits)
	return parseUint64FromHex(hash[:16])
}

// parseUint64FromHex converts a hex string to uint64
func parseUint64FromHex(hexStr string) (uint64, error) {
	var result uint64
	_, err := fmt.Sscanf(hexStr, "%x", &result)
	if err != nil {
		return 0, err
	}
	return result, nil
}
