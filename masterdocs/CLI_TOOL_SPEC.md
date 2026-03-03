# NovaCron CLI Tool Specification

## Overview

The NovaCron CLI (`nova`) provides a powerful, intuitive command-line interface for managing virtual machines, orchestrating deployments, and interacting with the NovaCron platform.

## Design Principles

1. **Intuitive**: Commands follow kubectl/docker patterns for familiarity
2. **Powerful**: Full API coverage with advanced features
3. **Interactive**: Rich terminal UI for complex operations
4. **Scriptable**: JSON/YAML output for automation
5. **Extensible**: Plugin system for custom commands

## Installation

```bash
# macOS
brew install novacron/tap/nova

# Linux
curl -L https://get.novacron.io | bash

# Windows
scoop install nova

# From source
go install github.com/novacron/cli/cmd/nova@latest
```

## Command Structure

```
nova [global-flags] <resource> <verb> [resource-name] [flags]
```

## Core Commands

### VM Management

```bash
# List all VMs
nova vm list
nova vm ls  # alias

# List VMs with filters
nova vm list --state running --node node-01
nova vm list --label app=web --output json

# Get VM details
nova vm get web-server-01
nova vm describe web-server-01  # detailed view

# Create VM
nova vm create web-server --image ubuntu-22.04 --cpu 4 --memory 8G --disk 100G
nova vm create -f vm-config.yaml

# Interactive VM creation
nova vm create --interactive

# Start/Stop/Restart VM
nova vm start web-server-01
nova vm stop web-server-01 --graceful --timeout 300
nova vm restart web-server-01

# Delete VM
nova vm delete web-server-01
nova vm delete web-server-01 --force --no-backup

# SSH into VM
nova vm ssh web-server-01
nova vm ssh web-server-01 --user admin --key ~/.ssh/custom_key

# Execute command in VM
nova vm exec web-server-01 -- ls -la /var/log
nova vm exec web-server-01 --script ./maintenance.sh

# Copy files to/from VM
nova vm cp local-file.txt web-server-01:/tmp/
nova vm cp web-server-01:/var/log/app.log ./logs/

# VM Console
nova vm console web-server-01
nova vm console web-server-01 --vnc  # Open VNC viewer

# Live Migration
nova vm migrate web-server-01 --target-node node-02
nova vm migrate web-server-01 --auto  # Auto-select best node

# Snapshots
nova vm snapshot create web-server-01 --name pre-upgrade
nova vm snapshot list web-server-01
nova vm snapshot restore web-server-01 pre-upgrade
nova vm snapshot delete web-server-01 pre-upgrade

# Resize VM
nova vm resize web-server-01 --cpu 8 --memory 16G
nova vm resize web-server-01 --disk +50G  # Add 50G to disk

# Monitor VM
nova vm stats web-server-01
nova vm stats web-server-01 --follow --interval 5s
nova vm top  # Show all VMs resource usage

# VM Templates
nova vm export web-server-01 --output template.yaml
nova vm import template.yaml --name new-server
```

### Pool Management (Auto-scaling Groups)

```bash
# List pools
nova pool list

# Create pool
nova pool create web-pool --replicas 3 --template web-server-template
nova pool create web-pool -f pool-config.yaml

# Scale pool
nova pool scale web-pool --replicas 5
nova pool scale web-pool --min 2 --max 10  # Set auto-scaling limits

# Update pool
nova pool update web-pool --image ubuntu-22.04-v2
nova pool rollout status web-pool
nova pool rollout history web-pool
nova pool rollout undo web-pool

# Delete pool
nova pool delete web-pool
```

### Network Management

```bash
# List networks
nova network list

# Create network
nova network create prod-net --subnet 10.0.0.0/24 --gateway 10.0.0.1
nova network create prod-net --type overlay --driver vxlan

# Connect/Disconnect VM to network
nova network connect prod-net web-server-01 --ip 10.0.0.10
nova network disconnect prod-net web-server-01

# Network policies
nova network policy create allow-web --network prod-net --port 80,443 --source 0.0.0.0/0
nova network policy list prod-net
nova network policy delete allow-web

# Network diagnostics
nova network inspect prod-net
nova network trace web-server-01 database-01  # Trace network path
```

### Storage Management

```bash
# List storage
nova storage list

# Create volume
nova storage create data-vol --size 100G --type ssd
nova storage create data-vol --from-snapshot snap-123

# Attach/Detach volume
nova storage attach data-vol web-server-01 --mount /data
nova storage detach data-vol web-server-01

# Expand volume
nova storage resize data-vol --size 200G

# Storage snapshots
nova storage snapshot create data-vol --name backup-$(date +%Y%m%d)
nova storage snapshot list data-vol

# Storage metrics
nova storage stats data-vol
```

### Cluster Management

```bash
# Cluster info
nova cluster info
nova cluster status

# Node management
nova node list
nova node get node-01
nova node drain node-01 --grace-period 300
nova node cordon node-01  # Mark as unschedulable
nova node uncordon node-01

# Join/Remove nodes
nova node join --token abc123 --master 10.0.0.1:6443
nova node remove node-05 --force

# Cluster health
nova cluster health
nova cluster diagnose  # Run diagnostic tests
```

### Configuration Management

```bash
# Config contexts (multiple clusters)
nova config get-contexts
nova config use-context production
nova config current-context

# Add cluster
nova config add-cluster staging --server https://staging.novacron.io --insecure-skip-tls-verify

# Authentication
nova auth login --username admin
nova auth login --token $NOVA_TOKEN
nova auth login --oidc --provider google
nova auth logout

# Config view/edit
nova config view
nova config set defaults.cpu 2
nova config set defaults.memory 4G
```

### Monitoring & Logs

```bash
# Logs
nova logs web-server-01
nova logs web-server-01 --follow --tail 100
nova logs web-server-01 --since 1h --grep ERROR

# Events
nova events
nova events --watch
nova events --resource vm/web-server-01

# Metrics
nova metrics vm web-server-01 --metric cpu --last 1h
nova metrics cluster --format prometheus

# Alerts
nova alert list
nova alert create cpu-high --threshold "cpu > 80" --duration 5m --action email:ops@example.com
nova alert silence cpu-high --duration 2h --reason "Maintenance"
```

### Advanced Features

```bash
# Backup & Restore
nova backup create web-server-01 --destination s3://backups/
nova backup list
nova backup restore web-server-01 --from backup-20240120

# Cost Analysis
nova cost report --period month
nova cost forecast --period quarter
nova cost optimize --suggest  # Get optimization suggestions

# Compliance
nova compliance check --standard hipaa
nova compliance report --format pdf --output compliance-report.pdf

# Marketplace
nova marketplace search nginx
nova marketplace install nginx-optimized --name web-proxy
nova marketplace publish my-template --price 0.10

# GitOps
nova gitops init --repo https://github.com/org/infra
nova gitops sync
nova gitops diff  # Show pending changes
```

## Interactive Mode

```bash
# Enter interactive shell
nova shell

# Interactive mode commands
nova> vm list
nova> vm start web-server-01
nova> exit

# Interactive VM creation wizard
nova vm create --interactive
? VM Name: web-server-01
? Select image: ubuntu-22.04
? CPU cores: 4
? Memory: 8G
? Disk size: 100G
? Network: prod-net
? Start immediately? Yes
✓ VM created successfully
```

## Output Formats

```bash
# Table (default)
nova vm list

# JSON
nova vm list -o json
nova vm list --output json

# YAML
nova vm list -o yaml

# Custom columns
nova vm list -o custom-columns=NAME:.metadata.name,CPU:.spec.cpu,STATE:.status.state

# JSONPath
nova vm get web-server-01 -o jsonpath='{.status.ipAddress}'

# Go template
nova vm list -o go-template='{{range .items}}{{.metadata.name}} {{.status.state}}{{"\n"}}{{end}}'

# Wide (more columns)
nova vm list -o wide

# Name only
nova vm list -o name
```

## Configuration File

```yaml
# ~/.nova/config.yaml
current-context: production
contexts:
  - name: production
    cluster: prod-cluster
    user: admin
    namespace: default
  - name: staging
    cluster: staging-cluster
    user: developer
    namespace: staging

clusters:
  - name: prod-cluster
    server: https://api.novacron.io
    certificate-authority: /path/to/ca.crt
  - name: staging-cluster
    server: https://staging.novacron.io
    insecure-skip-tls-verify: true

users:
  - name: admin
    token: eyJhbGciOiJIUzI1NiIs...
  - name: developer
    client-certificate: /path/to/client.crt
    client-key: /path/to/client.key

defaults:
  cpu: 2
  memory: 4G
  disk: 50G
  image: ubuntu-22.04
  output: table
  
preferences:
  colors: true
  pager: less
  editor: vim
  confirm-destructive: true
```

## Plugin System

```bash
# List plugins
nova plugin list

# Install plugin
nova plugin install https://github.com/user/nova-plugin-backup
nova plugin install nova-plugin-terraform

# Use plugin
nova backup create web-server-01  # Uses backup plugin
nova terraform generate  # Uses terraform plugin

# Develop plugin
nova plugin scaffold my-plugin
```

### Plugin Structure

```go
// pkg/plugin/plugin.go
package plugin

import (
    "github.com/novacron/cli/pkg/plugin"
    "github.com/spf13/cobra"
)

type MyPlugin struct{}

func (p *MyPlugin) Name() string {
    return "my-plugin"
}

func (p *MyPlugin) Version() string {
    return "1.0.0"
}

func (p *MyPlugin) Commands() []*cobra.Command {
    return []*cobra.Command{
        {
            Use:   "my-command",
            Short: "My custom command",
            Run: func(cmd *cobra.Command, args []string) {
                // Implementation
            },
        },
    }
}

func init() {
    plugin.Register(&MyPlugin{})
}
```

## Shell Completion

```bash
# Bash
nova completion bash > /etc/bash_completion.d/nova

# Zsh
nova completion zsh > "${fpath[1]}/_nova"

# Fish
nova completion fish > ~/.config/fish/completions/nova.fish

# PowerShell
nova completion powershell > nova.ps1
```

## Environment Variables

```bash
# API endpoint
export NOVA_API_ENDPOINT=https://api.novacron.io

# Authentication
export NOVA_TOKEN=eyJhbGciOiJIUzI1NiIs...
export NOVA_USERNAME=admin
export NOVA_PASSWORD=secret

# Defaults
export NOVA_NAMESPACE=production
export NOVA_OUTPUT=json
export NOVA_NO_HEADERS=true

# Debugging
export NOVA_DEBUG=true
export NOVA_TRACE=true
export NOVA_LOG_LEVEL=debug
```

## Error Handling

```bash
# Verbose error output
nova vm create web-server --debug

# Dry run (don't actually execute)
nova vm delete web-server-01 --dry-run

# Force operation
nova vm delete web-server-01 --force

# Retry with backoff
nova vm start web-server-01 --retry 3 --retry-interval 10s

# Timeout
nova vm migrate web-server-01 --timeout 600s
```

## Scripting Examples

```bash
#!/bin/bash
# Automated VM provisioning

# Create VMs from list
for name in web-{01..10}; do
    nova vm create $name \
        --image ubuntu-22.04 \
        --cpu 4 \
        --memory 8G \
        --disk 100G \
        --network prod-net \
        --label tier=web \
        --wait
done

# Health check all VMs
nova vm list -o json | jq -r '.items[].metadata.name' | while read vm; do
    if ! nova vm health $vm --quiet; then
        echo "VM $vm is unhealthy, restarting..."
        nova vm restart $vm
    fi
done

# Backup all VMs
nova vm list -o name | xargs -P 4 -I {} nova backup create {} --async

# Scale based on load
load=$(nova metrics cluster --metric cpu --last 5m --aggregate avg -o json | jq -r '.value')
if (( $(echo "$load > 80" | bc -l) )); then
    current=$(nova pool get web-pool -o jsonpath='{.spec.replicas}')
    nova pool scale web-pool --replicas $((current + 2))
fi
```

## Implementation Architecture

```go
// cmd/nova/main.go
package main

import (
    "os"
    
    "github.com/novacron/cli/internal/cmd"
    "github.com/novacron/cli/pkg/client"
)

func main() {
    // Initialize client
    config, err := client.LoadConfig()
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
        os.Exit(1)
    }
    
    apiClient := client.NewClient(config)
    
    // Create root command
    rootCmd := cmd.NewRootCommand(apiClient)
    
    // Execute
    if err := rootCmd.Execute(); err != nil {
        os.Exit(1)
    }
}
```

```go
// internal/cmd/vm.go
package cmd

import (
    "fmt"
    
    "github.com/spf13/cobra"
    "github.com/novacron/cli/pkg/api"
    "github.com/novacron/cli/pkg/printer"
)

func NewVMCommand(client api.Client) *cobra.Command {
    cmd := &cobra.Command{
        Use:     "vm",
        Short:   "Manage virtual machines",
        Aliases: []string{"virtualmachine", "vms"},
    }
    
    // Subcommands
    cmd.AddCommand(NewVMListCommand(client))
    cmd.AddCommand(NewVMCreateCommand(client))
    cmd.AddCommand(NewVMDeleteCommand(client))
    // ... more subcommands
    
    return cmd
}

func NewVMListCommand(client api.Client) *cobra.Command {
    var output string
    var labels []string
    
    cmd := &cobra.Command{
        Use:   "list",
        Short: "List virtual machines",
        RunE: func(cmd *cobra.Command, args []string) error {
            // Get VMs from API
            vms, err := client.VMs().List(api.ListOptions{
                Labels: labels,
            })
            if err != nil {
                return fmt.Errorf("failed to list VMs: %w", err)
            }
            
            // Print results
            p := printer.GetPrinter(output)
            return p.Print(vms)
        },
    }
    
    cmd.Flags().StringVarP(&output, "output", "o", "table", "Output format")
    cmd.Flags().StringSliceVar(&labels, "label", []string{}, "Filter by labels")
    
    return cmd
}
```

## Testing

```bash
# Run CLI tests
nova test

# Test VM creation
nova test vm create --dry-run

# Integration tests
nova test integration --cluster test-cluster

# Performance tests
nova test performance --duration 10m --concurrent 100
```

## Documentation

```bash
# Built-in help
nova help
nova vm help
nova vm create --help

# Man pages
man nova
man nova-vm-create

# Generate documentation
nova doc generate --format markdown --output ./docs/cli
nova doc generate --format man --output /usr/local/man/man1

# Examples
nova examples vm
nova examples network
nova examples scripting
```