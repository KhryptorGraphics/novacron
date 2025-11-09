package flamegraph

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

// Generator creates flamegraphs from profile data
type Generator struct {
	config GeneratorConfig
}

// GeneratorConfig defines flamegraph generation parameters
type GeneratorConfig struct {
	OutputDir        string
	InteractiveHTML  bool
	DiffEnabled      bool
	HotspotThreshold float64 // 0.05 (5%)
	MinSamples       int64   // 10
	ColorScheme      string  // "hot", "mem", "io"
}

// Flamegraph represents a flamegraph
type Flamegraph struct {
	Name      string
	ProfilePath string
	SVGPath   string
	HTMLPath  string
	Hotspots  []Hotspot
	CallStacks []CallStack
	TotalSamples int64
}

// Hotspot represents a performance hotspot
type Hotspot struct {
	FunctionName string
	File         string
	Line         int
	Samples      int64
	Percentage   float64
	CallStack    string
}

// CallStack represents a call stack
type CallStack struct {
	Stack      []string
	Samples    int64
	Percentage float64
}

// NewGenerator creates flamegraph generator
func NewGenerator(config GeneratorConfig) *Generator {
	if config.HotspotThreshold == 0 {
		config.HotspotThreshold = 0.05
	}
	if config.MinSamples == 0 {
		config.MinSamples = 10
	}
	if config.ColorScheme == "" {
		config.ColorScheme = "hot"
	}

	return &Generator{
		config: config,
	}
}

// Generate creates flamegraph from profile
func (g *Generator) Generate(profilePath string) (*Flamegraph, error) {
	// Create output directory
	if err := os.MkdirAll(g.config.OutputDir, 0755); err != nil {
		return nil, fmt.Errorf("create output dir: %w", err)
	}

	baseName := filepath.Base(profilePath)
	baseName = strings.TrimSuffix(baseName, filepath.Ext(baseName))

	// Convert profile to collapsed stacks
	stacksPath := filepath.Join(g.config.OutputDir, baseName+".stacks")
	if err := g.convertToStacks(profilePath, stacksPath); err != nil {
		return nil, fmt.Errorf("convert to stacks: %w", err)
	}

	// Generate SVG flamegraph
	svgPath := filepath.Join(g.config.OutputDir, baseName+".svg")
	if err := g.generateSVG(stacksPath, svgPath); err != nil {
		return nil, fmt.Errorf("generate svg: %w", err)
	}

	// Parse stacks for analysis
	callStacks, totalSamples, err := g.parseStacks(stacksPath)
	if err != nil {
		return nil, fmt.Errorf("parse stacks: %w", err)
	}

	// Identify hotspots
	hotspots := g.identifyHotspots(callStacks, totalSamples)

	fg := &Flamegraph{
		Name:         baseName,
		ProfilePath:  profilePath,
		SVGPath:      svgPath,
		Hotspots:     hotspots,
		CallStacks:   callStacks,
		TotalSamples: totalSamples,
	}

	// Generate interactive HTML if enabled
	if g.config.InteractiveHTML {
		htmlPath := filepath.Join(g.config.OutputDir, baseName+".html")
		if err := g.generateHTML(fg, htmlPath); err != nil {
			return nil, fmt.Errorf("generate html: %w", err)
		}
		fg.HTMLPath = htmlPath
	}

	return fg, nil
}

// convertToStacks converts pprof to collapsed stacks
func (g *Generator) convertToStacks(profilePath, outputPath string) error {
	// Use go tool pprof to convert
	cmd := exec.Command("go", "tool", "pprof", "-raw", profilePath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("pprof conversion failed: %w, output: %s", err, output)
	}

	// Parse and collapse stacks
	stacks := g.collapseStacks(string(output))

	// Write collapsed stacks
	return os.WriteFile(outputPath, []byte(stacks), 0644)
}

// collapseStacks converts pprof output to collapsed format
func (g *Generator) collapseStacks(pprofOutput string) string {
	var collapsed strings.Builder

	scanner := bufio.NewScanner(strings.NewReader(pprofOutput))
	var currentStack []string
	var samples int64

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse sample count
		if strings.Contains(line, "samples/count") {
			fmt.Sscanf(line, "%d", &samples)
			continue
		}

		// Parse stack frame
		if strings.Contains(line, ":") {
			parts := strings.SplitN(line, " ", 2)
			if len(parts) > 0 {
				currentStack = append(currentStack, parts[0])
			}
		}

		// End of stack
		if line == "---" && len(currentStack) > 0 {
			// Reverse stack (root first)
			for i := len(currentStack)/2 - 1; i >= 0; i-- {
				opp := len(currentStack) - 1 - i
				currentStack[i], currentStack[opp] = currentStack[opp], currentStack[i]
			}

			collapsed.WriteString(strings.Join(currentStack, ";"))
			collapsed.WriteString(fmt.Sprintf(" %d\n", samples))

			currentStack = nil
			samples = 0
		}
	}

	return collapsed.String()
}

// generateSVG creates SVG flamegraph
func (g *Generator) generateSVG(stacksPath, svgPath string) error {
	// Read stacks
	stacksData, err := os.ReadFile(stacksPath)
	if err != nil {
		return err
	}

	// Generate simple SVG (in production, use flamegraph.pl or similar)
	svg := g.createSVGFromStacks(string(stacksData))

	return os.WriteFile(svgPath, []byte(svg), 0644)
}

// createSVGFromStacks creates basic SVG representation
func (g *Generator) createSVGFromStacks(stacks string) string {
	// Simplified SVG generation
	// In production, use proper flamegraph library
	var svg strings.Builder

	svg.WriteString(`<?xml version="1.0" standalone="no"?>`)
	svg.WriteString(`<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">`)
	svg.WriteString(`<svg version="1.1" width="1200" height="800" xmlns="http://www.w3.org/2000/svg">`)
	svg.WriteString(`<text x="600" y="30" text-anchor="middle" font-size="20">Performance Flamegraph</text>`)

	// Parse and render stacks
	lines := strings.Split(stacks, "\n")
	y := 50
	for _, line := range lines {
		if line == "" {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) < 2 {
			continue
		}

		stack := parts[0]
		samples := parts[1]

		svg.WriteString(fmt.Sprintf(`<rect x="10" y="%d" width="1180" height="20" fill="#%s"/>`,
			y, g.getColor(stack)))
		svg.WriteString(fmt.Sprintf(`<text x="15" y="%d" font-size="12">%s (%s)</text>`,
			y+15, stack, samples))
		y += 25
	}

	svg.WriteString(`</svg>`)
	return svg.String()
}

// getColor returns color based on function name
func (g *Generator) getColor(function string) string {
	// Simple color scheme based on name hash
	hash := 0
	for _, c := range function {
		hash = hash*31 + int(c)
	}
	hash = hash % 360

	// HSL to RGB conversion for nice colors
	return fmt.Sprintf("%06x", hash*65536 + 128*256 + 128)
}

// parseStacks parses collapsed stacks
func (g *Generator) parseStacks(stacksPath string) ([]CallStack, int64, error) {
	data, err := os.ReadFile(stacksPath)
	if err != nil {
		return nil, 0, err
	}

	var callStacks []CallStack
	var totalSamples int64

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}

		stack := strings.Split(parts[0], ";")
		var samples int64
		fmt.Sscanf(parts[1], "%d", &samples)

		totalSamples += samples

		callStacks = append(callStacks, CallStack{
			Stack:      stack,
			Samples:    samples,
			Percentage: 0, // Will calculate later
		})
	}

	// Calculate percentages
	for i := range callStacks {
		callStacks[i].Percentage = float64(callStacks[i].Samples) / float64(totalSamples)
	}

	return callStacks, totalSamples, nil
}

// identifyHotspots identifies performance hotspots
func (g *Generator) identifyHotspots(callStacks []CallStack, totalSamples int64) []Hotspot {
	functionSamples := make(map[string]int64)

	// Aggregate samples by function
	for _, cs := range callStacks {
		for _, fn := range cs.Stack {
			functionSamples[fn] += cs.Samples
		}
	}

	// Find hotspots above threshold
	var hotspots []Hotspot
	for fn, samples := range functionSamples {
		if samples < g.config.MinSamples {
			continue
		}

		percentage := float64(samples) / float64(totalSamples)
		if percentage >= g.config.HotspotThreshold {
			hotspots = append(hotspots, Hotspot{
				FunctionName: fn,
				Samples:      samples,
				Percentage:   percentage,
			})
		}
	}

	// Sort by percentage
	sort.Slice(hotspots, func(i, j int) bool {
		return hotspots[i].Percentage > hotspots[j].Percentage
	})

	return hotspots
}

// generateHTML creates interactive HTML viewer
func (g *Generator) generateHTML(fg *Flamegraph, htmlPath string) error {
	var html bytes.Buffer

	html.WriteString(`<!DOCTYPE html><html><head><title>Flamegraph: ` + fg.Name + `</title>`)
	html.WriteString(`<style>body{font-family:Arial,sans-serif;margin:20px;}`)
	html.WriteString(`.hotspot{background:#fff3cd;padding:10px;margin:5px;border-left:3px solid #ffc107;}`)
	html.WriteString(`</style></head><body>`)
	html.WriteString(`<h1>Flamegraph: ` + fg.Name + `</h1>`)
	html.WriteString(fmt.Sprintf(`<p>Total Samples: %d</p>`, fg.TotalSamples))

	// Embed SVG
	svgData, _ := os.ReadFile(fg.SVGPath)
	html.Write(svgData)

	// Hotspots section
	html.WriteString(`<h2>Performance Hotspots</h2>`)
	for _, hs := range fg.Hotspots {
		html.WriteString(fmt.Sprintf(`<div class="hotspot"><strong>%s</strong>: %.2f%% (%d samples)</div>`,
			hs.FunctionName, hs.Percentage*100, hs.Samples))
	}

	html.WriteString(`</body></html>`)

	return os.WriteFile(htmlPath, html.Bytes(), 0644)
}

// DiffFlamegraphs compares two flamegraphs
func (g *Generator) DiffFlamegraphs(before, after *Flamegraph) (*FlamegraphDiff, error) {
	if !g.config.DiffEnabled {
		return nil, fmt.Errorf("diff not enabled")
	}

	diff := &FlamegraphDiff{
		Before: before.Name,
		After:  after.Name,
	}

	// Compare hotspots
	beforeHotspots := make(map[string]float64)
	for _, hs := range before.Hotspots {
		beforeHotspots[hs.FunctionName] = hs.Percentage
	}

	for _, hs := range after.Hotspots {
		beforePct := beforeHotspots[hs.FunctionName]
		change := hs.Percentage - beforePct

		diff.Changes = append(diff.Changes, HotspotChange{
			FunctionName:     hs.FunctionName,
			BeforePercentage: beforePct,
			AfterPercentage:  hs.Percentage,
			Change:           change,
			ChangePercent:    (change / beforePct) * 100,
		})
	}

	// Sort by absolute change
	sort.Slice(diff.Changes, func(i, j int) bool {
		return abs(diff.Changes[i].Change) > abs(diff.Changes[j].Change)
	})

	return diff, nil
}

// FlamegraphDiff represents difference between flamegraphs
type FlamegraphDiff struct {
	Before  string
	After   string
	Changes []HotspotChange
}

// HotspotChange represents change in hotspot
type HotspotChange struct {
	FunctionName     string
	BeforePercentage float64
	AfterPercentage  float64
	Change           float64
	ChangePercent    float64
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
