# AI Engine Database Configuration

## Overview

The NovaCron AI Engine modules now support configurable SQLite database paths for production deployment. This allows for proper data persistence, separation of concerns, and compliance with system administration best practices.

## Configuration Methods

Database paths can be configured using the following methods (in order of priority):

### 1. Direct Parameter (Highest Priority)

Pass the database path directly when instantiating the classes:

```python
from ai_engine.predictive_scaling import PredictiveScalingEngine
from ai_engine.workload_pattern_recognition import WorkloadPatternRecognizer
from ai_engine.performance_optimizer import PerformancePredictor

# Use custom paths
ps_engine = PredictiveScalingEngine(db_path="/opt/novacron/data/scaling.db")
wpr_engine = WorkloadPatternRecognizer(db_path="/opt/novacron/data/patterns.db")
perf_predictor = PerformancePredictor(db_path="/opt/novacron/data/performance.db")
```

### 2. Individual Environment Variables

Set specific environment variables for each module:

```bash
export PREDICTIVE_SCALING_DB="/var/lib/novacron/predictive_scaling.db"
export WORKLOAD_PATTERNS_DB="/var/lib/novacron/workload_patterns.db"
export PERFORMANCE_DB="/var/lib/novacron/performance.db"
```

### 3. Common Data Directory

Set a single environment variable for all NovaCron data:

```bash
export NOVACRON_DATA_DIR="/var/lib/novacron"
```

This will automatically create the following database files:
- `$NOVACRON_DATA_DIR/predictive_scaling.db`
- `$NOVACRON_DATA_DIR/workload_patterns.db`
- `$NOVACRON_DATA_DIR/performance_predictor.db`

### 4. Default Location

If no configuration is provided, databases will be created in:
- `/var/lib/novacron/` (default production location)

### 5. Automatic Fallback

If the configured directory is not writable, the system will automatically fall back to `/tmp/` with a warning logged.

## Production Deployment

### Recommended Setup

1. **Create dedicated data directory:**
```bash
sudo mkdir -p /var/lib/novacron
sudo chown novacron:novacron /var/lib/novacron
sudo chmod 750 /var/lib/novacron
```

2. **Set environment in systemd service:**
```ini
[Service]
Environment="NOVACRON_DATA_DIR=/var/lib/novacron"
```

3. **Or use Docker volumes:**
```yaml
volumes:
  - /var/lib/novacron:/var/lib/novacron
environment:
  - NOVACRON_DATA_DIR=/var/lib/novacron
```

### Directory Permissions

The AI Engine will:
1. Attempt to create the directory if it doesn't exist
2. Test write permissions before using the directory
3. Fall back to `/tmp/` if the directory is not writable
4. Log all database path decisions for troubleshooting

## Testing Configuration

Run the included test script to verify your configuration:

```bash
python tests/test_ai_db_config.py
```

This will test:
- Default path behavior
- Environment variable handling
- Directory creation and permissions
- Fallback mechanisms
- Direct parameter overrides

## Logging

Database path selection is logged at startup:

```
INFO: Using database path: /var/lib/novacron/predictive_scaling.db
```

Or if falling back:

```
WARNING: Cannot write to /var/lib/novacron: Permission denied. Falling back to /tmp
INFO: Using fallback database path: /tmp/predictive_scaling.db
```

## Migration from /tmp

If you have existing data in `/tmp/`, you can migrate it:

```bash
# Stop NovaCron services
sudo systemctl stop novacron

# Create new data directory
sudo mkdir -p /var/lib/novacron
sudo chown novacron:novacron /var/lib/novacron

# Copy existing databases
sudo cp /tmp/predictive_scaling.db /var/lib/novacron/
sudo cp /tmp/workload_patterns.db /var/lib/novacron/
sudo cp /tmp/performance_predictor.db /var/lib/novacron/
sudo chown novacron:novacron /var/lib/novacron/*.db

# Update environment configuration
echo "NOVACRON_DATA_DIR=/var/lib/novacron" >> /etc/novacron/novacron.env

# Restart services
sudo systemctl start novacron
```

## Troubleshooting

### Permission Denied Errors

If you see permission errors, ensure:
1. The directory exists
2. The user running NovaCron has write permissions
3. SELinux/AppArmor policies allow database access

### Database Lock Errors

SQLite databases can be locked if:
1. Multiple processes access the same database
2. The filesystem doesn't support proper locking (some network filesystems)
3. Solution: Use separate database files per process or implement connection pooling

### Disk Space Issues

Monitor disk usage in your data directory:
```bash
du -sh /var/lib/novacron/*.db
```

Consider implementing rotation or cleanup policies for historical data.

## Best Practices

1. **Production Systems:** Always use a persistent directory like `/var/lib/novacron`
2. **Development:** Can use `/tmp` or project-specific directories
3. **Docker/Kubernetes:** Mount volumes for database persistence
4. **Backups:** Include database files in backup procedures
5. **Monitoring:** Monitor database file sizes and query performance
6. **Security:** Ensure proper file permissions (640 or 600)

## Module-Specific Databases

### Predictive Scaling Database
- **File:** `predictive_scaling.db`
- **Purpose:** Stores scaling history, resource forecasts, cost tracking, and model performance metrics
- **Typical Size:** 10-100 MB depending on history retention

### Workload Pattern Recognition Database
- **File:** `workload_patterns.db`
- **Purpose:** Stores workload patterns, classification history, and pattern evolution data
- **Typical Size:** 5-50 MB depending on pattern diversity

### Performance Predictor Database
- **File:** `performance_predictor.db`
- **Purpose:** Stores performance metrics, predictions, and optimization history
- **Typical Size:** 20-200 MB depending on metric granularity