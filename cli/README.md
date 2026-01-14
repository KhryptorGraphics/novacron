# NovaCron CLI

Command-line interface for managing virtual machines in a distributed NovaCron cluster.

## Features

- **VM Lifecycle Management**: Create, start, stop, restart, and delete VMs
- **Live Migration**: Migrate VMs between nodes with minimal downtime
- **Resource Management**: Resize VM resources (CPU, memory, disk)
- **Monitoring**: View VM status, metrics, logs, and events
- **Interactive Console**: Connect to VM console for direct access
- **Snapshot Management**: Create and manage VM snapshots
- **Multi-Cluster Support**: Manage multiple NovaCron clusters
- **Multiple Output Formats**: Table, JSON, YAML, and wide formats
- **Shell Completion**: Bash, Zsh, Fish, and PowerShell completion
- **Plugin System**: Extend CLI functionality with plugins

## Installation

### From Source

```bash
go install github.com/novacron/cli/cmd/novacron@latest
```

### Pre-built Binaries

Download the latest release for your platform from the [releases page](https://github.com/novacron/cli/releases).

### Build from Source

```bash
git clone https://github.com/novacron/cli.git
cd cli
go build -o novacron cmd/novacron/main.go
sudo mv novacron /usr/local/bin/
```

## Configuration

### Initial Setup

```bash
# Configure cluster connection
novacron config set-cluster prod --server https://api.novacron.example.com

# Login to cluster
novacron login --cluster prod

# Set default cluster
novacron config use-context prod
```

### Configuration File

The CLI stores configuration in `~/.novacron/config.yaml`:

```yaml
currentCluster: prod
clusters:
  prod:
    name: prod
    server: https://api.novacron.example.com
    namespace: default
    insecure: false
    authType: token
preferences:
  output: table
  noColor: false
  verbose: false
  pageSize: 20
  timeout: 30
```

## Usage

### VM Management

```bash
# List all VMs
novacron vm list

# List VMs in all namespaces
novacron vm list --all-namespaces

# Get VM details
novacron vm get my-vm

# Create a VM
novacron vm create my-vm \
  --cpu 4 \
  --memory 8Gi \
  --disk 100Gi \
  --image marketplace://ubuntu-22.04-lts

# Create VM from file
novacron vm create -f vm.yaml

# Start/Stop/Restart VM
novacron vm start my-vm
novacron vm stop my-vm
novacron vm restart my-vm

# Delete VM
novacron vm delete my-vm

# Resize VM
novacron vm resize my-vm --cpu 8 --memory 16Gi

# Migrate VM
novacron vm migrate my-vm --target-node node2 --live
```

### Console Access

```bash
# Connect to VM console
novacron vm console my-vm

# Execute command in VM
novacron exec my-vm -- ls -la

# Copy files to/from VM
novacron copy local-file.txt my-vm:/tmp/
novacron copy my-vm:/var/log/app.log ./

# Port forwarding
novacron port-forward my-vm 8080:80
```

### Monitoring

```bash
# View VM logs
novacron logs my-vm
novacron logs my-vm --follow --tail 100

# Monitor resource usage
novacron top
novacron top vm

# Get VM events
novacron get events --field-selector involvedObject.name=my-vm

# View metrics
novacron monitor metrics my-vm
```

### Snapshot Management

```bash
# Create snapshot
novacron snapshot create my-vm-snap --vm my-vm

# List snapshots
novacron snapshot list

# Restore from snapshot
novacron snapshot restore my-vm-snap

# Delete snapshot
novacron snapshot delete my-vm-snap
```

### Cluster Management

```bash
# List nodes
novacron node list

# Get node details
novacron node get node1

# Drain node for maintenance
novacron node drain node1

# List clusters
novacron cluster list

# Get cluster info
novacron cluster info
```

### Output Formats

```bash
# Table format (default)
novacron vm list

# JSON format
novacron vm list -o json

# YAML format
novacron vm list -o yaml

# Wide format (more columns)
novacron vm list -o wide

# Custom columns
novacron get vm -o custom-columns=NAME:.metadata.name,STATUS:.status.phase
```

### Advanced Usage

```bash
# Apply configuration from file
novacron apply -f resources.yaml

# Watch for changes
novacron get vm --watch

# Filter by labels
novacron vm list -l app=web,tier=frontend

# Batch operations
novacron delete vm -l env=test

# Dry run
novacron create vm test-vm --dry-run -o yaml

# Export configuration
novacron get vm my-vm -o yaml > vm-backup.yaml
```

## Shell Completion

### Bash

```bash
# Add to ~/.bashrc
source <(novacron completion bash)

# Or save to file
novacron completion bash > /etc/bash_completion.d/novacron
```

### Zsh

```bash
# Add to ~/.zshrc
source <(novacron completion zsh)

# Or save to file
novacron completion zsh > "${fpath[1]}/_novacron"
```

### Fish

```bash
novacron completion fish | source

# Or save to file
novacron completion fish > ~/.config/fish/completions/novacron.fish
```

### PowerShell

```powershell
novacron completion powershell | Out-String | Invoke-Expression

# Or save to profile
novacron completion powershell > $PROFILE
```

## Plugin System

The CLI supports plugins to extend functionality:

```bash
# List installed plugins
novacron plugin list

# Install a plugin
novacron plugin install <plugin-name>

# Remove a plugin
novacron plugin remove <plugin-name>

# Update plugins
novacron plugin update
```

Plugins are discovered from:
- `~/.novacron/plugins/`
- `$PATH` (binaries prefixed with `novacron-`)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NOVACRON_CONFIG` | Config file path | `~/.novacron/config.yaml` |
| `NOVACRON_CLUSTER` | Default cluster | From config |
| `NOVACRON_NAMESPACE` | Default namespace | `default` |
| `NOVACRON_OUTPUT` | Output format | `table` |
| `NOVACRON_NO_COLOR` | Disable colors | `false` |
| `NOVACRON_VERBOSE` | Verbose output | `false` |

## Troubleshooting

### Enable Debug Output

```bash
novacron --verbose vm list
# or
export NOVACRON_VERBOSE=true
novacron vm list
```

### Check Configuration

```bash
novacron config view
novacron config get-clusters
```

### Test Connection

```bash
novacron cluster info
novacron version
```

### Common Issues

**Connection refused**
```bash
# Check server URL
novacron config get-cluster

# Test with insecure mode (development only)
novacron --insecure vm list
```

**Authentication failed**
```bash
# Re-login
novacron login --cluster prod

# Check token expiry
novacron auth info
```

**Timeout errors**
```bash
# Increase timeout
novacron --timeout 60 vm create large-vm
```

## Development

### Building

```bash
# Build binary
go build -o novacron cmd/novacron/main.go

# Run tests
go test ./...

# Generate mocks
go generate ./...
```

### Adding Commands

1. Create command file in `internal/commands/`
2. Implement command logic
3. Add to root command in `root.go`
4. Update completion scripts

### Plugin Development

Plugins must:
- Be executable files prefixed with `novacron-`
- Accept `--help` flag
- Return JSON when called with `--metadata`
- Handle standard CLI arguments

Example plugin structure:
```go
package main

import (
    "github.com/spf13/cobra"
    "github.com/novacron/cli/pkg/plugin"
)

func main() {
    plugin.Execute(&cobra.Command{
        Use:   "my-plugin",
        Short: "My custom plugin",
        RunE:  run,
    })
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Support

- Documentation: https://docs.novacron.io/cli
- Issues: https://github.com/novacron/cli/issues
- Discussions: https://github.com/novacron/cli/discussions
- Slack: https://novacron.slack.com/channels/cli