# DWCP Manager - Quick Reference Card

## Essential Commands
```bash
# Start/Stop/Restart
sudo systemctl start dwcp-manager
sudo systemctl stop dwcp-manager
sudo systemctl restart dwcp-manager

# Health Check
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics

# Circuit Breaker State
curl http://localhost:8080/circuit-breaker/state

# Logs
journalctl -u dwcp-manager -f
```

## Health Endpoints
- Health: `http://localhost:8080/health`
- Metrics: `http://localhost:8080/metrics`
- Circuit Breaker: `http://localhost:8080/circuit-breaker/state`

## Key Metrics
- `dwcp_manager_uptime_seconds` - Uptime
- `dwcp_transport_active_streams` - Active streams
- `dwcp_manager_circuit_breaker_state` - Circuit state (0=closed, 1=half-open, 2=open)

## Common Issues
| Issue | Quick Fix |
|-------|-----------|
| Manager won't start | Check `/etc/dwcp/config.yaml` syntax |
| Circuit breaker open | `curl -X POST http://localhost:8080/circuit-breaker/reset` |
| High memory | Reduce buffer sizes in config |
| Transport unhealthy | Disable RDMA if not available |

## Configuration Locations
- Config: `/etc/dwcp/config.yaml`
- Binary: `/opt/dwcp/bin/dwcp-manager`
- Logs: `/var/log/dwcp/manager.log`
- Data: `/var/lib/dwcp/`

## On-Call: ops-oncall@example.com
