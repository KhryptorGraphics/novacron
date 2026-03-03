# DWCP Phase 1 - Quick Deployment Reference

## 🚀 Quick Start (5 Minutes)

### Prerequisites Check
```bash
go version          # Must be 1.21+
curl http://localhost:9090/metrics  # Prometheus running
```

### Deploy to Staging
```bash
# One-line deployment
sudo ./scripts/deploy-dwcp-phase1.sh staging

# Validate
./scripts/validate-dwcp.sh
```

### Deploy via CI/CD
```bash
git checkout -b dwcp-phase1
git push origin dwcp-phase1
# GitHub Actions handles the rest
```

---

## 📋 Essential Commands

### Health Check
```bash
curl http://localhost:8080/health
```

### Check DWCP Status
```bash
curl http://localhost:8080/api/v1/dwcp/status
```

### View Metrics
```bash
curl http://localhost:9090/metrics | grep dwcp_
```

### Quick Validation
```bash
./scripts/validate-dwcp.sh
```

### Emergency Rollback
```bash
# Disable DWCP
sudo sed -i 's/enabled: true/enabled: false/' /etc/dwcp/dwcp.staging.yaml

# Restart
killall dwcp-api-server && /usr/local/bin/dwcp-api-server &
```

---

## 🔍 Key Metrics to Monitor

| Metric | Good Value | Action if Below |
|--------|------------|-----------------|
| `dwcp_amst_active_streams` | > 16 | Check network connectivity |
| `dwcp_hde_compression_ratio` | > 2.0 | Review compression config |
| `dwcp_error_rate` | < 5% | Check logs for errors |
| `dwcp_hde_baselines_synced` | > 0 | Wait for sync or restart |

---

## 🎯 Success Checklist

- [ ] Service starts successfully
- [ ] Health endpoint returns `"status": "healthy"`
- [ ] AMST streams > 0
- [ ] HDE compression ratio > 2.0
- [ ] Error rate < 5%
- [ ] Prometheus metrics available
- [ ] All validation checks pass

---

## 🆘 Troubleshooting Quick Reference

### Service won't start
```bash
# Check logs
tail -f /var/log/dwcp.log

# Verify config
yamllint /etc/dwcp/dwcp.staging.yaml

# Check port conflicts
sudo lsof -i :8080
```

### No active streams
```bash
# Check network
ping remote-host

# Verify AMST config
curl http://localhost:8080/api/v1/dwcp/amst/config
```

### Low compression
```bash
# Check HDE status
curl http://localhost:9090/metrics | grep hde_

# Verify baselines
curl http://localhost:9090/metrics | grep baselines_synced
```

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `/configs/dwcp.yaml` | Base configuration |
| `/configs/dwcp.staging.yaml` | Staging overrides |
| `/scripts/deploy-dwcp-phase1.sh` | Deployment automation |
| `/scripts/validate-dwcp.sh` | Validation tool |
| `/docs/DWCP-PHASE1-DEPLOYMENT.md` | Full documentation |

---

## 🔄 Rollback Steps

1. **Stop service**: `killall dwcp-api-server`
2. **Restore backup**: `cp $(cat /tmp/dwcp-latest-backup)/* /etc/dwcp/`
3. **Restart**: `/usr/local/bin/dwcp-api-server &`
4. **Validate**: `./scripts/validate-dwcp.sh`

---

## 📞 Getting Help

- Full documentation: `/docs/DWCP-PHASE1-DEPLOYMENT.md`
- Architecture docs: `/docs/architecture/distributed-wan-communication-protocol.md`
- CI/CD pipeline: `/.github/workflows/dwcp-phase1-deploy.yml`
