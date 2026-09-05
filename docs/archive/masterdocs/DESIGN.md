# NovaCron CLI Design Document

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Design](#component-design)
4. [API Design](#api-design)
5. [Command Structure](#command-structure)
6. [Plugin System](#plugin-system)
7. [Configuration Management](#configuration-management)
8. [Output System](#output-system)
9. [Error Handling](#error-handling)
10. [Testing Strategy](#testing-strategy)

## Overview

The NovaCron CLI (`nova`) is a powerful command-line interface designed to provide comprehensive management capabilities for the NovaCron distributed VM orchestration platform. It follows modern CLI design principles inspired by `kubectl`, `docker`, and `aws` CLI tools.

### Design Principles

1. **Intuitive**: Familiar patterns for users of kubectl/docker
2. **Composable**: Commands can be chained and scripted
3. **Extensible**: Plugin architecture for custom functionality
4. **Performant**: Efficient API calls with caching
5. **Interactive**: Rich terminal UI for complex operations
6. **Discoverable**: Comprehensive help and auto-completion

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Input                          │
└─────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    CLI Entry Point     │
                    │      (main.go)         │
                    └───────────┬───────────┘
                                │
                ┌───────────────▼───────────────┐
                │       Command Router          │
                │         (Cobra)               │
                └───────┬───────────────┬───────┘
                        │               │
            ┌───────────▼──────┐ ┌─────▼───────────┐
            │  Core Commands   │ │     Plugins      │
            └───────────┬──────┘ └─────┬───────────┘
                        │               │
                ┌───────▼───────────────▼───────┐
                │      Command Executor          │
                └───────────────┬───────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    Service Layer       │
                    ├───────────────────────┤
                    │ • VM Service          │
                    │ • Network Service     │
                    │ • Storage Service     │
                    │ • Cluster Service     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │     API Client         │
                    │  (REST/WebSocket)      │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    NovaCron API        │
                    └───────────────────────┘
```

### Component Layers

```
┌──────────────────────────────────────────────────────┐
│                  Presentation Layer                   │
│  • Command Interface  • Output Formatters  • UI      │
├──────────────────────────────────────────────────────┤
│                   Business Logic Layer                │
│  • Command Handlers  • Validation  • Transformation  │
├──────────────────────────────────────────────────────┤
│                    Service Layer                      │
│  • Resource Services  • Business Operations          │
├──────────────────────────────────────────────────────┤
│                  Infrastructure Layer                 │
│  • API Client  • Config Manager  • Cache  • Storage  │
└──────────────────────────────────────────────────────┘
```

## Component Design

### Core Components

#### 1. Command Router (`cmd/root.go`)

```go
package cmd

import (
    "github.com/spf13/cobra"
    "github.com/spf13/viper"
)

type RootCommand struct {
    config     *config.Config
    apiClient  *client.Client
    services   *services.ServiceRegistry
    plugins    *plugin.Manager
}

func NewRootCommand() *cobra.Command {
    root := &RootCommand{}
    
    cmd := &cobra.Command{
        Use:   "nova",
        Short: "NovaCron CLI - Distributed VM Orchestration",
        PersistentPreRunE: root.initialize,
    }
    
    // Add subcommands
    cmd.AddCommand(
        NewVMCommand(root.services),
        NewNetworkCommand(root.services),
        NewStorageCommand(root.services),
        NewClusterCommand(root.services),
        NewConfigCommand(root.config),
    )
    
    // Add global flags
    root.addGlobalFlags(cmd)
    
    return cmd
}
```

#### 2. API Client (`pkg/client/client.go`)

```go
package client

import (
    "context"
    "net/http"
    "time"
)

type Client struct {
    baseURL    string
    httpClient *http.Client
    websocket  *WebSocketClient
    auth       Authenticator
    cache      Cache
    
    // Resource clients
    VMs        VMClient
    Networks   NetworkClient
    Storage    StorageClient
    Clusters   ClusterClient
}

type ClientOptions struct {
    BaseURL     string
    Token       string
    Timeout     time.Duration
    RetryPolicy RetryPolicy
    CacheConfig CacheConfig
}

func NewClient(opts ClientOptions) (*Client, error) {
    // Implementation
}

// Request builder pattern
type RequestBuilder struct {
    client  *Client
    method  string
    path    string
    params  map[string]string
    body    interface{}
    headers map[string]string
}

func (c *Client) NewRequest(method, path string) *RequestBuilder {
    return &RequestBuilder{
        client: c,
        method: method,
        path:   path,
        params: make(map[string]string),
        headers: make(map[string]string),
    }
}
```

#### 3. Service Layer (`pkg/services/vm_service.go`)

```go
package services

import (
    "context"
    "github.com/novacron/cli/pkg/client"
    "github.com/novacron/cli/pkg/models"
)

type VMService interface {
    List(ctx context.Context, opts ListOptions) ([]*models.VM, error)
    Get(ctx context.Context, id string) (*models.VM, error)
    Create(ctx context.Context, spec *models.VMSpec) (*models.VM, error)
    Update(ctx context.Context, id string, spec *models.VMSpec) (*models.VM, error)
    Delete(ctx context.Context, id string) error
    
    // Operations
    Start(ctx context.Context, id string) error
    Stop(ctx context.Context, id string, graceful bool) error
    Restart(ctx context.Context, id string) error
    Migrate(ctx context.Context, id string, target string) error
    
    // Advanced
    Console(ctx context.Context, id string) (ConsoleSession, error)
    Exec(ctx context.Context, id string, command []string) (ExecSession, error)
    Copy(ctx context.Context, id string, src, dst string) error
    
    // Monitoring
    Stats(ctx context.Context, id string) (*models.VMStats, error)
    Events(ctx context.Context, id string) (EventStream, error)
    Logs(ctx context.Context, id string, opts LogOptions) (LogStream, error)
}

type vmService struct {
    client *client.Client
    cache  Cache
}

func NewVMService(client *client.Client) VMService {
    return &vmService{
        client: client,
        cache:  NewLRUCache(100),
    }
}
```

#### 4. Output System (`pkg/output/printer.go`)

```go
package output

import (
    "io"
    "encoding/json"
    "gopkg.in/yaml.v3"
    "github.com/olekukonko/tablewriter"
)

type Printer interface {
    Print(data interface{}) error
    PrintError(err error) error
}

type Format string

const (
    FormatTable      Format = "table"
    FormatJSON       Format = "json"
    FormatYAML       Format = "yaml"
    FormatWide       Format = "wide"
    FormatCustom     Format = "custom"
    FormatName       Format = "name"
    FormatGoTemplate Format = "go-template"
    FormatJSONPath   Format = "jsonpath"
)

type PrinterFactory struct {
    writer io.Writer
    format Format
    opts   PrintOptions
}

type PrintOptions struct {
    NoHeaders      bool
    ShowLabels     bool
    SortBy         string
    CustomColumns  []string
    Template       string
    JSONPath       string
}

func NewPrinterFactory(w io.Writer, format Format, opts PrintOptions) *PrinterFactory {
    return &PrinterFactory{
        writer: w,
        format: format,
        opts:   opts,
    }
}

func (f *PrinterFactory) GetPrinter() Printer {
    switch f.format {
    case FormatJSON:
        return &JSONPrinter{writer: f.writer}
    case FormatYAML:
        return &YAMLPrinter{writer: f.writer}
    case FormatTable:
        return &TablePrinter{writer: f.writer, opts: f.opts}
    case FormatGoTemplate:
        return &TemplatePrinter{writer: f.writer, template: f.opts.Template}
    case FormatJSONPath:
        return &JSONPathPrinter{writer: f.writer, path: f.opts.JSONPath}
    default:
        return &TablePrinter{writer: f.writer, opts: f.opts}
    }
}
```

## API Design

### RESTful API Client Interface

```go
package api

// Resource interfaces follow RESTful patterns
type ResourceClient[T any] interface {
    List(ctx context.Context, opts ListOptions) (*List[T], error)
    Get(ctx context.Context, name string, opts GetOptions) (*T, error)
    Create(ctx context.Context, obj *T, opts CreateOptions) (*T, error)
    Update(ctx context.Context, obj *T, opts UpdateOptions) (*T, error)
    Patch(ctx context.Context, name string, patch Patch, opts PatchOptions) (*T, error)
    Delete(ctx context.Context, name string, opts DeleteOptions) error
    Watch(ctx context.Context, opts WatchOptions) (Watch[T], error)
}

// Common options
type ListOptions struct {
    LabelSelector string
    FieldSelector string
    Limit         int
    Continue      string
    TimeoutSeconds *int
}

type GetOptions struct {
    ResourceVersion string
}

type CreateOptions struct {
    DryRun []string
}

type UpdateOptions struct {
    DryRun []string
    FieldManager string
}

type DeleteOptions struct {
    GracePeriodSeconds *int64
    PropagationPolicy  *PropagationPolicy
    DryRun            []string
}

type WatchOptions struct {
    LabelSelector     string
    FieldSelector     string
    Watch            bool
    ResourceVersion  string
    TimeoutSeconds   *int64
}
```

### WebSocket Client

```go
package client

import (
    "github.com/gorilla/websocket"
)

type WebSocketClient struct {
    url    string
    conn   *websocket.Conn
    events chan Event
    errors chan error
}

type Event struct {
    Type   EventType   `json:"type"`
    Object interface{} `json:"object"`
}

type EventType string

const (
    EventAdded    EventType = "ADDED"
    EventModified EventType = "MODIFIED"
    EventDeleted  EventType = "DELETED"
    EventError    EventType = "ERROR"
)

func (ws *WebSocketClient) Connect() error {
    // Establish WebSocket connection
}

func (ws *WebSocketClient) Subscribe(resource string, opts SubscribeOptions) error {
    // Subscribe to resource events
}

func (ws *WebSocketClient) Events() <-chan Event {
    return ws.events
}
```

## Command Structure

### Command Hierarchy

```
nova
├── vm (virtualmachine, vms)
│   ├── list (ls)
│   ├── get (describe)
│   ├── create
│   ├── update (edit)
│   ├── delete (rm)
│   ├── start
│   ├── stop
│   ├── restart
│   ├── migrate
│   ├── console
│   ├── exec
│   ├── copy (cp)
│   ├── snapshot
│   ├── resize
│   ├── stats
│   ├── logs
│   └── events
├── pool
│   ├── list
│   ├── create
│   ├── scale
│   ├── update
│   └── delete
├── network
│   ├── list
│   ├── create
│   ├── connect
│   ├── disconnect
│   ├── policy
│   └── inspect
├── storage
│   ├── list
│   ├── create
│   ├── attach
│   ├── detach
│   ├── resize
│   └── snapshot
├── cluster
│   ├── info
│   ├── status
│   ├── health
│   └── diagnose
├── node
│   ├── list
│   ├── get
│   ├── drain
│   ├── cordon
│   └── uncordon
├── config
│   ├── view
│   ├── set
│   ├── get-contexts
│   ├── use-context
│   └── current-context
├── auth
│   ├── login
│   └── logout
├── plugin
│   ├── list
│   ├── install
│   └── uninstall
├── completion
│   ├── bash
│   ├── zsh
│   ├── fish
│   └── powershell
└── help
```

### Command Implementation Pattern

```go
package cmd

import (
    "github.com/spf13/cobra"
    "github.com/novacron/cli/pkg/services"
)

type VMCommand struct {
    service services.VMService
    printer output.Printer
}

func NewVMCommand(service services.VMService) *cobra.Command {
    vc := &VMCommand{service: service}
    
    cmd := &cobra.Command{
        Use:     "vm",
        Short:   "Manage virtual machines",
        Aliases: []string{"virtualmachine", "vms"},
    }
    
    // Subcommands
    cmd.AddCommand(
        vc.newListCommand(),
        vc.newGetCommand(),
        vc.newCreateCommand(),
        vc.newDeleteCommand(),
        vc.newStartCommand(),
        vc.newStopCommand(),
        // ... more subcommands
    )
    
    return cmd
}

func (vc *VMCommand) newListCommand() *cobra.Command {
    var (
        output  string
        labels  []string
        state   string
        limit   int
    )
    
    cmd := &cobra.Command{
        Use:     "list",
        Short:   "List virtual machines",
        Aliases: []string{"ls"},
        RunE: func(cmd *cobra.Command, args []string) error {
            ctx := cmd.Context()
            
            opts := services.ListOptions{
                Labels: labels,
                State:  state,
                Limit:  limit,
            }
            
            vms, err := vc.service.List(ctx, opts)
            if err != nil {
                return err
            }
            
            printer := output.GetPrinter(output)
            return printer.Print(vms)
        },
    }
    
    cmd.Flags().StringVarP(&output, "output", "o", "table", "Output format")
    cmd.Flags().StringSliceVar(&labels, "label", []string{}, "Filter by labels")
    cmd.Flags().StringVar(&state, "state", "", "Filter by state")
    cmd.Flags().IntVar(&limit, "limit", 0, "Maximum number of results")
    
    return cmd
}
```

## Plugin System

### Plugin Architecture

```go
package plugin

import (
    "plugin"
    "github.com/spf13/cobra"
)

type Plugin interface {
    Name() string
    Version() string
    Commands() []*cobra.Command
    Initialize(config *Config) error
}

type Manager struct {
    plugins map[string]Plugin
    loader  Loader
    config  *Config
}

type Loader interface {
    Load(path string) (Plugin, error)
    Discover(dir string) ([]string, error)
}

type GoPluginLoader struct{}

func (l *GoPluginLoader) Load(path string) (Plugin, error) {
    p, err := plugin.Open(path)
    if err != nil {
        return nil, err
    }
    
    symbol, err := p.Lookup("Plugin")
    if err != nil {
        return nil, err
    }
    
    plugin, ok := symbol.(Plugin)
    if !ok {
        return nil, fmt.Errorf("invalid plugin type")
    }
    
    return plugin, nil
}

// Plugin implementation example
type BackupPlugin struct {
    client *client.Client
}

func (p *BackupPlugin) Name() string {
    return "backup"
}

func (p *BackupPlugin) Version() string {
    return "1.0.0"
}

func (p *BackupPlugin) Commands() []*cobra.Command {
    return []*cobra.Command{
        {
            Use:   "backup",
            Short: "Backup virtual machines",
            Run:   p.backup,
        },
        {
            Use:   "restore",
            Short: "Restore virtual machines",
            Run:   p.restore,
        },
    }
}
```

## Configuration Management

### Configuration Structure

```go
package config

import (
    "github.com/spf13/viper"
)

type Config struct {
    CurrentContext string              `yaml:"current-context"`
    Contexts       []Context           `yaml:"contexts"`
    Clusters       []Cluster           `yaml:"clusters"`
    Users          []User              `yaml:"users"`
    Defaults       Defaults            `yaml:"defaults"`
    Preferences    Preferences         `yaml:"preferences"`
}

type Context struct {
    Name      string `yaml:"name"`
    Cluster   string `yaml:"cluster"`
    User      string `yaml:"user"`
    Namespace string `yaml:"namespace"`
}

type Cluster struct {
    Name                  string `yaml:"name"`
    Server                string `yaml:"server"`
    CertificateAuthority  string `yaml:"certificate-authority,omitempty"`
    InsecureSkipTLSVerify bool   `yaml:"insecure-skip-tls-verify,omitempty"`
}

type User struct {
    Name              string `yaml:"name"`
    Token             string `yaml:"token,omitempty"`
    ClientCertificate string `yaml:"client-certificate,omitempty"`
    ClientKey         string `yaml:"client-key,omitempty"`
}

type Defaults struct {
    CPU    int    `yaml:"cpu"`
    Memory string `yaml:"memory"`
    Disk   string `yaml:"disk"`
    Image  string `yaml:"image"`
    Output string `yaml:"output"`
}

type Preferences struct {
    Colors              bool   `yaml:"colors"`
    Pager               string `yaml:"pager"`
    Editor              string `yaml:"editor"`
    ConfirmDestructive  bool   `yaml:"confirm-destructive"`
}

type Manager struct {
    config *Config
    viper  *viper.Viper
    path   string
}

func NewManager() (*Manager, error) {
    m := &Manager{
        viper: viper.New(),
        path:  getConfigPath(),
    }
    
    if err := m.Load(); err != nil {
        return nil, err
    }
    
    return m, nil
}

func (m *Manager) GetCurrentContext() (*Context, error) {
    for _, ctx := range m.config.Contexts {
        if ctx.Name == m.config.CurrentContext {
            return &ctx, nil
        }
    }
    return nil, fmt.Errorf("current context not found")
}
```

## Output System

### Interactive Mode

```go
package interactive

import (
    "github.com/manifoldco/promptui"
    "github.com/charmbracelet/bubbles/table"
    tea "github.com/charmbracelet/bubbletea"
)

type InteractiveMode struct {
    program *tea.Program
}

func NewInteractiveMode() *InteractiveMode {
    return &InteractiveMode{}
}

func (im *InteractiveMode) VMCreateWizard() (*VMSpec, error) {
    spec := &VMSpec{}
    
    // Name prompt
    namePrompt := promptui.Prompt{
        Label: "VM Name",
        Validate: func(input string) error {
            if len(input) < 3 {
                return fmt.Errorf("name must be at least 3 characters")
            }
            return nil
        },
    }
    spec.Name, _ = namePrompt.Run()
    
    // Image selection
    imageSelect := promptui.Select{
        Label: "Select Image",
        Items: []string{
            "ubuntu-22.04",
            "ubuntu-20.04",
            "centos-8",
            "debian-11",
            "windows-server-2022",
        },
    }
    _, spec.Image, _ = imageSelect.Run()
    
    // CPU selection
    cpuSelect := promptui.Select{
        Label: "CPU Cores",
        Items: []string{"1", "2", "4", "8", "16"},
    }
    _, cpuStr, _ := cpuSelect.Run()
    spec.CPU, _ = strconv.Atoi(cpuStr)
    
    // Memory selection
    memorySelect := promptui.Select{
        Label: "Memory",
        Items: []string{"1G", "2G", "4G", "8G", "16G", "32G"},
    }
    _, spec.Memory, _ = memorySelect.Run()
    
    // Disk size
    diskPrompt := promptui.Prompt{
        Label:   "Disk Size",
        Default: "50G",
    }
    spec.Disk, _ = diskPrompt.Run()
    
    // Network selection
    networkSelect := promptui.Select{
        Label: "Network",
        Items: []string{"default", "production", "development"},
    }
    _, spec.Network, _ = networkSelect.Run()
    
    // Confirmation
    confirmPrompt := promptui.Prompt{
        Label:     "Create VM with these settings?",
        IsConfirm: true,
    }
    result, _ := confirmPrompt.Run()
    
    if result != "y" {
        return nil, fmt.Errorf("cancelled")
    }
    
    return spec, nil
}
```

### Progress Indicators

```go
package progress

import (
    "github.com/vbauerster/mpb/v7"
    "github.com/vbauerster/mpb/v7/decor"
)

type ProgressTracker struct {
    progress *mpb.Progress
    bars     map[string]*mpb.Bar
}

func NewProgressTracker() *ProgressTracker {
    return &ProgressTracker{
        progress: mpb.New(),
        bars:     make(map[string]*mpb.Bar),
    }
}

func (pt *ProgressTracker) AddBar(name string, total int64) *mpb.Bar {
    bar := pt.progress.AddBar(total,
        mpb.PrependDecorators(
            decor.Name(name, decor.WC{W: len(name) + 1, C: decor.DidentRight}),
            decor.CountersNoUnit("%d / %d", decor.WCSyncWidth),
        ),
        mpb.AppendDecorators(
            decor.Percentage(decor.WC{W: 5}),
            decor.OnComplete(
                decor.EwmaETA(decor.ET_STYLE_GO, 60),
                "done",
            ),
        ),
    )
    
    pt.bars[name] = bar
    return bar
}

func (pt *ProgressTracker) UpdateBar(name string, current int64) {
    if bar, ok := pt.bars[name]; ok {
        bar.SetCurrent(current)
    }
}

func (pt *ProgressTracker) Wait() {
    pt.progress.Wait()
}
```

## Error Handling

### Error Types

```go
package errors

import (
    "fmt"
)

type ErrorType string

const (
    ErrorTypeValidation   ErrorType = "ValidationError"
    ErrorTypeNotFound     ErrorType = "NotFound"
    ErrorTypeConflict     ErrorType = "Conflict"
    ErrorTypeUnauthorized ErrorType = "Unauthorized"
    ErrorTypeForbidden    ErrorType = "Forbidden"
    ErrorTypeInternal     ErrorType = "InternalError"
    ErrorTypeTimeout      ErrorType = "Timeout"
    ErrorTypeNetwork      ErrorType = "NetworkError"
)

type CLIError struct {
    Type    ErrorType
    Message string
    Details string
    Cause   error
}

func (e *CLIError) Error() string {
    if e.Details != "" {
        return fmt.Sprintf("%s: %s (%s)", e.Type, e.Message, e.Details)
    }
    return fmt.Sprintf("%s: %s", e.Type, e.Message)
}

func (e *CLIError) Unwrap() error {
    return e.Cause
}

// Helper functions
func NewValidationError(message string) *CLIError {
    return &CLIError{
        Type:    ErrorTypeValidation,
        Message: message,
    }
}

func NewNotFoundError(resource, name string) *CLIError {
    return &CLIError{
        Type:    ErrorTypeNotFound,
        Message: fmt.Sprintf("%s '%s' not found", resource, name),
    }
}

// Error handler
type ErrorHandler struct {
    verbose bool
    debug   bool
}

func (h *ErrorHandler) Handle(err error) {
    if err == nil {
        return
    }
    
    switch e := err.(type) {
    case *CLIError:
        h.handleCLIError(e)
    default:
        h.handleGenericError(err)
    }
}

func (h *ErrorHandler) handleCLIError(err *CLIError) {
    fmt.Fprintf(os.Stderr, "Error: %s\n", err.Message)
    
    if h.verbose && err.Details != "" {
        fmt.Fprintf(os.Stderr, "Details: %s\n", err.Details)
    }
    
    if h.debug && err.Cause != nil {
        fmt.Fprintf(os.Stderr, "Cause: %v\n", err.Cause)
    }
    
    // Suggest actions based on error type
    switch err.Type {
    case ErrorTypeUnauthorized:
        fmt.Fprintf(os.Stderr, "Hint: Try running 'nova auth login'\n")
    case ErrorTypeNotFound:
        fmt.Fprintf(os.Stderr, "Hint: Use 'nova vm list' to see available VMs\n")
    case ErrorTypeNetwork:
        fmt.Fprintf(os.Stderr, "Hint: Check your network connection and API endpoint\n")
    }
}
```

## Testing Strategy

### Unit Tests

```go
package cmd_test

import (
    "testing"
    "bytes"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
)

type MockVMService struct {
    mock.Mock
}

func (m *MockVMService) List(ctx context.Context, opts ListOptions) ([]*VM, error) {
    args := m.Called(ctx, opts)
    return args.Get(0).([]*VM), args.Error(1)
}

func TestVMListCommand(t *testing.T) {
    // Setup
    mockService := new(MockVMService)
    mockService.On("List", mock.Anything, mock.Anything).Return([]*VM{
        {ID: "vm-1", Name: "test-vm-1"},
        {ID: "vm-2", Name: "test-vm-2"},
    }, nil)
    
    cmd := NewVMCommand(mockService)
    buf := new(bytes.Buffer)
    cmd.SetOut(buf)
    cmd.SetArgs([]string{"list", "--output", "json"})
    
    // Execute
    err := cmd.Execute()
    
    // Assert
    assert.NoError(t, err)
    assert.Contains(t, buf.String(), "test-vm-1")
    assert.Contains(t, buf.String(), "test-vm-2")
    mockService.AssertExpectations(t)
}
```

### Integration Tests

```go
package integration_test

import (
    "testing"
    "os/exec"
)

func TestCLIIntegration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }
    
    // Build CLI
    cmd := exec.Command("go", "build", "-o", "nova-test", "./cmd/nova")
    if err := cmd.Run(); err != nil {
        t.Fatalf("Failed to build CLI: %v", err)
    }
    
    // Test VM list
    cmd = exec.Command("./nova-test", "vm", "list")
    output, err := cmd.CombinedOutput()
    if err != nil {
        t.Fatalf("Failed to run vm list: %v", err)
    }
    
    // Verify output
    if !bytes.Contains(output, []byte("NAME")) {
        t.Errorf("Expected header in output, got: %s", output)
    }
}
```

## Performance Considerations

### Caching Strategy

```go
package cache

import (
    "time"
    "sync"
)

type Cache interface {
    Get(key string) (interface{}, bool)
    Set(key string, value interface{}, ttl time.Duration)
    Delete(key string)
    Clear()
}

type LRUCache struct {
    mu       sync.RWMutex
    items    map[string]*cacheItem
    capacity int
    order    []string
}

type cacheItem struct {
    value     interface{}
    expiresAt time.Time
}

func NewLRUCache(capacity int) *LRUCache {
    return &LRUCache{
        items:    make(map[string]*cacheItem),
        capacity: capacity,
        order:    make([]string, 0, capacity),
    }
}

func (c *LRUCache) Get(key string) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    item, ok := c.items[key]
    if !ok {
        return nil, false
    }
    
    if time.Now().After(item.expiresAt) {
        c.Delete(key)
        return nil, false
    }
    
    // Move to front (MRU)
    c.moveToFront(key)
    
    return item.value, true
}
```

### Parallel Operations

```go
package parallel

import (
    "context"
    "sync"
    "golang.org/x/sync/errgroup"
)

type ParallelExecutor struct {
    maxConcurrency int
}

func NewParallelExecutor(maxConcurrency int) *ParallelExecutor {
    return &ParallelExecutor{
        maxConcurrency: maxConcurrency,
    }
}

func (pe *ParallelExecutor) Execute(ctx context.Context, tasks []Task) error {
    g, ctx := errgroup.WithContext(ctx)
    
    // Semaphore for concurrency control
    sem := make(chan struct{}, pe.maxConcurrency)
    
    for _, task := range tasks {
        task := task // Capture loop variable
        
        g.Go(func() error {
            sem <- struct{}{}
            defer func() { <-sem }()
            
            return task.Execute(ctx)
        })
    }
    
    return g.Wait()
}

type Task interface {
    Execute(ctx context.Context) error
}
```

## Security Considerations

### Authentication

```go
package auth

import (
    "github.com/golang-jwt/jwt/v4"
)

type Authenticator interface {
    Login(ctx context.Context, credentials Credentials) (*Token, error)
    Refresh(ctx context.Context, token string) (*Token, error)
    Logout(ctx context.Context) error
    GetToken() (string, error)
}

type TokenAuthenticator struct {
    store TokenStore
}

type Token struct {
    AccessToken  string    `json:"access_token"`
    RefreshToken string    `json:"refresh_token"`
    ExpiresAt    time.Time `json:"expires_at"`
    Type         string    `json:"token_type"`
}

type TokenStore interface {
    Save(token *Token) error
    Load() (*Token, error)
    Delete() error
}

type SecureTokenStore struct {
    keyring keyring.Keyring
}

func (s *SecureTokenStore) Save(token *Token) error {
    data, err := json.Marshal(token)
    if err != nil {
        return err
    }
    
    return s.keyring.Set(keyring.Item{
        Key:   "novacron-token",
        Data:  data,
        Label: "NovaCron CLI Token",
    })
}
```

## Extensibility Points

### Custom Resource Types

```go
package extensibility

// ResourceHandler allows plugins to register custom resource types
type ResourceHandler interface {
    Kind() string
    List(ctx context.Context) ([]interface{}, error)
    Get(ctx context.Context, name string) (interface{}, error)
    Create(ctx context.Context, obj interface{}) error
    Update(ctx context.Context, obj interface{}) error
    Delete(ctx context.Context, name string) error
}

// Registry manages custom resource handlers
type Registry struct {
    handlers map[string]ResourceHandler
}

func (r *Registry) Register(handler ResourceHandler) {
    r.handlers[handler.Kind()] = handler
}

func (r *Registry) GetHandler(kind string) (ResourceHandler, bool) {
    handler, ok := r.handlers[kind]
    return handler, ok
}
```

## Conclusion

This design provides a comprehensive, extensible, and user-friendly CLI for the NovaCron platform. The architecture supports:

1. **Modularity**: Clean separation of concerns with distinct layers
2. **Extensibility**: Plugin system for custom functionality
3. **Performance**: Caching and parallel execution strategies
4. **Usability**: Multiple output formats and interactive mode
5. **Maintainability**: Clear interfaces and testing strategies
6. **Security**: Secure token storage and authentication

The design follows industry best practices and provides a solid foundation for building a production-ready CLI tool.