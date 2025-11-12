# MCP Server Configuration Export

This directory contains scripts and configuration files to replicate the MCP server setup from this Claude Code instance to another server.

## Quick Setup

### On the New Server:

```bash
# 1. Copy these files to the new server
scp scripts/setup-mcp-servers.sh user@newserver:/path/to/destination/
scp scripts/mcp-servers-list.json user@newserver:/path/to/destination/

# 2. On the new server, make the script executable
chmod +x setup-mcp-servers.sh

# 3. Run the setup script
./setup-mcp-servers.sh
```

## Files Included

### `setup-mcp-servers.sh`
Interactive bash script that:
- Installs all MCP servers with prompts for optional ones
- Provides authentication instructions
- Verifies installation
- Shows post-installation steps

### `mcp-servers-list.json`
JSON configuration file containing:
- Complete list of MCP servers
- Installation commands
- Purpose and documentation links
- Authentication requirements
- Post-installation steps

### `README-MCP-SETUP.md`
This file - documentation for the setup process

## MCP Servers Configuration

### Required
- **claude-flow**: Core orchestration, SPARC workflows, 54 agents
  ```bash
  claude mcp add claude-flow npx claude-flow@alpha mcp start
  ```

### Optional
- **ruv-swarm**: Enhanced coordination
  ```bash
  claude mcp add ruv-swarm npx ruv-swarm mcp start
  ```

- **flow-nexus**: Cloud features (requires auth)
  ```bash
  claude mcp add flow-nexus npx flow-nexus@latest mcp start
  npx flow-nexus@latest login
  ```

- **beads**: Issue tracking
  ```bash
  npm install -g @beads/beads-cli
  bd init  # In your project directory
  ```

- **deep-research**: Web research and documentation
  ```bash
  # Check package documentation for installation
  ```

- **ide**: Built-in VS Code integration (no installation needed)

## Manual Installation

If you prefer manual installation without the script:

```bash
# Required
claude mcp add claude-flow npx claude-flow@alpha mcp start

# Optional - Add as needed
claude mcp add ruv-swarm npx ruv-swarm mcp start
claude mcp add flow-nexus npx flow-nexus@latest mcp start

# Verify
claude mcp list
claude mcp status
```

## Project Files to Copy

Don't forget to copy these project-specific files:

1. **CLAUDE.md** - Project instructions and agent documentation
2. **.claude/** - Custom slash commands
3. **package.json** - NPM scripts and dependencies
4. **.beads/** - Issue tracking database (if using beads)

## Authentication

### Flow Nexus (if installed)
```bash
# Register new account
npx flow-nexus@latest register

# Or login with existing account
npx flow-nexus@latest login
```

### Beads (if installed)
```bash
# Initialize in project directory
cd /path/to/project
bd init
```

## Verification

After installation:

```bash
# Check installed servers
claude mcp list

# Check server status
claude mcp status

# Test in Claude Code
# Try commands like:
# /flow-nexus:swarm
# /beads:list
# npx claude-flow sparc modes
```

## Troubleshooting

### MCP Server Not Found
- Ensure Node.js and npm are installed
- Check network connectivity for npx downloads
- Verify Claude Code is updated to latest version

### Authentication Issues
- For flow-nexus: Re-run `npx flow-nexus@latest login`
- Check credentials are correct
- Verify internet connection

### Beads Database Issues
- Run `bd init` in project directory
- Check `.beads/` directory has correct permissions
- Use `bd where-am-i` to verify context

### Command Not Working
- Restart Claude Code after installing MCP servers
- Run `claude mcp status` to check server health
- Check server logs for errors

## Additional Resources

- **Claude Flow**: https://github.com/ruvnet/claude-flow
- **Flow Nexus**: https://flow-nexus.ruv.io
- **Claude Code Docs**: https://docs.claude.com/claude-code
- **MCP Specification**: https://modelcontextprotocol.io

## Notes

- claude-flow is **REQUIRED** for core SPARC and swarm functionality
- All other servers are optional based on your needs
- MCP servers can be added/removed anytime with `claude mcp add/remove`
- Some features in CLAUDE.md require specific MCP servers
- Check `mcp-servers-list.json` for complete configuration details

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review server documentation links
3. Verify all prerequisites are installed
4. Check Claude Code version compatibility
