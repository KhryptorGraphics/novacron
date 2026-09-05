# DWCP Compression Telemetry Data Collection Plan

**Target**: 100,000+ labeled samples for training compression selector
**Timeline**: 4 weeks (2 weeks staging + 2 weeks production shadow mode)
**Status**: Ready for implementation

---

## 1. Data Schema

### PostgreSQL Table Definition

```sql
CREATE TABLE compression_telemetry (
    -- Primary key
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Request metadata
    vm_id VARCHAR(64) NOT NULL,
    request_id VARCHAR(128) NOT NULL,
    session_id VARCHAR(128),

    -- Network context (6 features)
    link_type VARCHAR(20) NOT NULL CHECK (link_type IN ('dc', 'metro', 'wan', 'edge')),
    region VARCHAR(20) NOT NULL,
    network_tier INTEGER NOT NULL CHECK (network_tier BETWEEN 1 AND 3),
    bandwidth_available_mbps FLOAT NOT NULL CHECK (bandwidth_available_mbps > 0),
    bandwidth_utilization FLOAT NOT NULL CHECK (bandwidth_utilization BETWEEN 0 AND 100),
    rtt_ms FLOAT NOT NULL CHECK (rtt_ms >= 0),

    -- Payload characteristics (5 features)
    payload_size BIGINT NOT NULL CHECK (payload_size > 0),
    payload_type VARCHAR(50) NOT NULL,
    entropy_estimate FLOAT CHECK (entropy_estimate BETWEEN 0 AND 8),
    repetition_score FLOAT CHECK (repetition_score BETWEEN 0 AND 100),
    has_baseline BOOLEAN NOT NULL,

    -- HDE metrics (4 features)
    hde_compression_ratio FLOAT CHECK (hde_compression_ratio > 0),
    hde_delta_hit_rate FLOAT CHECK (hde_delta_hit_rate BETWEEN 0 AND 100),
    hde_latency_ms FLOAT CHECK (hde_latency_ms >= 0),
    hde_cpu_usage FLOAT CHECK (hde_cpu_usage BETWEEN 0 AND 100),

    -- AMST metrics (4 features)
    amst_stream_count INTEGER CHECK (amst_stream_count > 0),
    amst_transfer_rate_mbps FLOAT CHECK (amst_transfer_rate_mbps >= 0),
    amst_latency_ms FLOAT CHECK (amst_latency_ms >= 0),
    amst_cpu_usage FLOAT CHECK (amst_cpu_usage BETWEEN 0 AND 100),

    -- Ground truth oracle label
    oracle_algorithm VARCHAR(10) NOT NULL CHECK (oracle_algorithm IN ('HDE', 'AMST')),
    oracle_utility_score FLOAT NOT NULL,
    oracle_computation_time_ms FLOAT,

    -- Actual algorithm used (for shadow mode validation)
    actual_algorithm VARCHAR(10),
    actual_latency_ms FLOAT,
    actual_throughput_mbps FLOAT,

    -- Data quality flags
    is_training_candidate BOOLEAN DEFAULT TRUE,
    quality_score FLOAT DEFAULT 1.0,

    -- Indexes for efficient querying
    INDEX idx_timestamp (timestamp DESC),
    INDEX idx_vm_id (vm_id),
    INDEX idx_link_region (link_type, region),
    INDEX idx_oracle (oracle_algorithm),
    INDEX idx_training (is_training_candidate) WHERE is_training_candidate = TRUE
);

-- Partitioning by month for scalability
CREATE TABLE compression_telemetry_202511 PARTITION OF compression_telemetry
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

-- Auto-generate future partitions
CREATE EXTENSION IF NOT EXISTS pg_partman;
SELECT create_parent('public.compression_telemetry', 'timestamp', 'native', 'monthly');
```

---

## 2. Integration Points (Go Implementation)

### File: `backend/core/network/dwcp/compression/telemetry_collector.go`

```go
package compression

import (
    "context"
    "database/sql"
    "time"
    "math"

    _ "github.com/lib/pq"
)

// TelemetryCollector collects compression metrics for ML training
type TelemetryCollector struct {
    db         *sql.DB
    config     *CollectorConfig
    insertStmt *sql.Stmt
}

type CollectorConfig struct {
    Enabled         bool
    DatabaseDSN     string
    BatchSize       int
    FlushInterval   time.Duration
    SamplingRate    float64  // 0.0-1.0 (default 1.0 = collect all)
}

type TelemetryRecord struct {
    Timestamp      time.Time
    VMID           string
    RequestID      string
    SessionID      string

    // Network features
    LinkType       string
    Region         string
    NetworkTier    int
    BandwidthMbps  float64
    BandwidthUtil  float64
    RTTms          float64

    // Payload features
    PayloadSize    int64
    PayloadType    string
    Entropy        float64
    RepetitionPct  float64
    HasBaseline    bool

    // HDE metrics
    HDECompressionRatio float64
    HDEDeltaHitRate     float64
    HDELatencyMs        float64
    HDECPUUsage         float64

    // AMST metrics
    AMSTStreamCount     int
    AMSTTransferRate    float64
    AMSTLatencyMs       float64
    AMSTCPUUsage        float64

    // Oracle decision
    OracleAlgorithm     string
    OracleUtilityScore  float64
}

func NewTelemetryCollector(config *CollectorConfig) (*TelemetryCollector, error) {
    if !config.Enabled {
        return nil, nil
    }

    db, err := sql.Open("postgres", config.DatabaseDSN)
    if err != nil {
        return nil, err
    }

    // Prepare insert statement
    stmt, err := db.Prepare(`
        INSERT INTO compression_telemetry (
            timestamp, vm_id, request_id, session_id,
            link_type, region, network_tier,
            bandwidth_available_mbps, bandwidth_utilization, rtt_ms,
            payload_size, payload_type, entropy_estimate,
            repetition_score, has_baseline,
            hde_compression_ratio, hde_delta_hit_rate,
            hde_latency_ms, hde_cpu_usage,
            amst_stream_count, amst_transfer_rate_mbps,
            amst_latency_ms, amst_cpu_usage,
            oracle_algorithm, oracle_utility_score
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19,
            $20, $21, $22, $23, $24, $25
        )
    `)
    if err != nil {
        return nil, err
    }

    return &TelemetryCollector{
        db:         db,
        config:     config,
        insertStmt: stmt,
    }, nil
}

func (tc *TelemetryCollector) CollectTelemetry(ctx context.Context, record TelemetryRecord) error {
    // Sampling (optional)
    if tc.config.SamplingRate < 1.0 {
        if math.Float64frombits(uint64(time.Now().UnixNano())) > tc.config.SamplingRate {
            return nil  // Skip this sample
        }
    }

    // Async insert to avoid blocking main thread
    go func() {
        _, err := tc.insertStmt.ExecContext(ctx,
            record.Timestamp, record.VMID, record.RequestID, record.SessionID,
            record.LinkType, record.Region, record.NetworkTier,
            record.BandwidthMbps, record.BandwidthUtil, record.RTTms,
            record.PayloadSize, record.PayloadType, record.Entropy,
            record.RepetitionPct, record.HasBaseline,
            record.HDECompressionRatio, record.HDEDeltaHitRate,
            record.HDELatencyMs, record.HDECPUUsage,
            record.AMSTStreamCount, record.AMSTTransferRate,
            record.AMSTLatencyMs, record.AMSTCPUUsage,
            record.OracleAlgorithm, record.OracleUtilityScore,
        )
        if err != nil {
            // Log error but don't fail main operation
            log.Printf("Telemetry insert failed: %v", err)
        }
    }()

    return nil
}

func (tc *TelemetryCollector) Close() error {
    if tc.insertStmt != nil {
        tc.insertStmt.Close()
    }
    if tc.db != nil {
        return tc.db.Close()
    }
    return nil
}

// Helper: Compute offline oracle algorithm selection
func ComputeOfflineOracle(features TelemetryRecord) (string, float64) {
    // HDE utility: (compression_ratio * bandwidth) / (latency * cpu_cost)
    hdeUtility := (features.HDECompressionRatio * features.BandwidthMbps) /
                  (features.HDELatencyMs * features.HDECPUUsage * 0.01 + 1e-6)

    // AMST utility: (transfer_rate * bandwidth) / (latency * cpu_cost)
    amstUtility := (features.AMSTTransferRate * features.BandwidthMbps * 0.01) /
                   (features.AMSTLatencyMs * features.AMSTCPUUsage * 0.01 + 1e-6)

    if hdeUtility > amstUtility {
        return "HDE", hdeUtility
    }
    return "AMST", amstUtility
}
```

---

## 3. Integration Hook in DWCP Manager

### File: `backend/core/network/dwcp/dwcp_manager.go` (modification)

```go
// Add to DWCPManager struct
type DWCPManager struct {
    // ... existing fields ...

    telemetryCollector *compression.TelemetryCollector
}

// In CompressAndTransfer method (after compression operation)
func (dm *DWCPManager) CompressAndTransfer(
    ctx context.Context,
    vmID string,
    data []byte,
) error {
    // ... existing compression logic ...

    // Collect telemetry (if enabled)
    if dm.telemetryCollector != nil {
        record := compression.TelemetryRecord{
            Timestamp:   time.Now(),
            VMID:        vmID,
            RequestID:   generateRequestID(),
            SessionID:   dm.sessionID,

            // Extract network features
            LinkType:    dm.detectLinkType(),
            Region:      dm.config.Region,
            NetworkTier: dm.config.NetworkTier,
            // ... populate all fields ...
        }

        // Compute oracle label
        record.OracleAlgorithm, record.OracleUtilityScore =
            compression.ComputeOfflineOracle(record)

        dm.telemetryCollector.CollectTelemetry(ctx, record)
    }

    return nil
}
```

---

## 4. Data Export for Training

### Export Script: `scripts/export_training_data.sh`

```bash
#!/bin/bash
# Export compression telemetry to CSV for ML training

OUTPUT_FILE="data/dwcp_training_$(date +%Y%m%d).csv"

psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "
COPY (
    SELECT
        timestamp,
        link_type,
        region,
        network_tier,
        bandwidth_available_mbps,
        bandwidth_utilization,
        rtt_ms,
        payload_size,
        payload_type,
        entropy_estimate,
        repetition_score,
        has_baseline,
        hde_compression_ratio,
        hde_delta_hit_rate,
        hde_latency_ms,
        hde_cpu_usage,
        amst_stream_count,
        amst_transfer_rate_mbps,
        amst_latency_ms,
        amst_cpu_usage,
        oracle_algorithm
    FROM compression_telemetry
    WHERE is_training_candidate = TRUE
      AND quality_score > 0.8
      AND timestamp >= NOW() - INTERVAL '30 days'
    ORDER BY timestamp DESC
    LIMIT 100000
) TO STDOUT WITH CSV HEADER
" > $OUTPUT_FILE

echo "Exported $(wc -l < $OUTPUT_FILE) records to $OUTPUT_FILE"
```

---

## 5. Data Quality Validation

### Validation Script: `scripts/validate_training_data.py`

```python
#!/usr/bin/env python3
"""Validate data quality before training"""

import pandas as pd
import numpy as np

def validate_training_data(csv_path):
    df = pd.read_csv(csv_path)

    print(f"Total records: {len(df)}")
    print(f"\nOracle distribution:")
    print(df['oracle_algorithm'].value_counts())

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Missing values detected:")
        print(missing[missing > 0])

    # Check for outliers
    for col in ['bandwidth_available_mbps', 'rtt_ms', 'payload_size']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
        if outliers > 0:
            print(f"⚠️  {col}: {outliers} outliers detected")

    # Check class balance
    class_balance = df['oracle_algorithm'].value_counts(normalize=True)
    if (class_balance < 0.3).any() or (class_balance > 0.7).any():
        print(f"⚠️  Class imbalance detected: {class_balance}")

    print(f"\n✅ Validation complete")

if __name__ == '__main__':
    import sys
    validate_training_data(sys.argv[1])
```

---

## 6. Timeline & Milestones

| Week | Phase | Activities | Deliverables |
|------|-------|-----------|--------------|
| 1 | Setup | Deploy telemetry collector to staging | PostgreSQL table, Go integration |
| 2 | Staging | Collect 10K samples, validate schema | Data quality report |
| 3 | Production | Shadow mode deployment (50% traffic) | 50K samples |
| 4 | Production | Full telemetry collection (100% traffic) | 100K+ samples |
| 5 | Export | Data export, validation, oracle labeling | Training-ready CSV |
| 6 | Training | Model training with v3 script | Trained model (≥98% accuracy) |

---

## 7. Deployment Checklist

- [ ] PostgreSQL database provisioned
- [ ] Telemetry table created with partitioning
- [ ] Go telemetry collector implemented
- [ ] Integration hooks added to dwcp_manager.go
- [ ] Configuration flag for enabling/disabling telemetry
- [ ] Monitoring dashboard for telemetry collection rate
- [ ] Export script tested
- [ ] Validation script tested
- [ ] Privacy review completed (ensure no PII in telemetry)
- [ ] Load testing (telemetry should not impact performance)

---

## 8. Privacy & Security Considerations

1. **No PII**: VM IDs should be anonymized/hashed
2. **Data retention**: Telemetry older than 90 days auto-deleted
3. **Access control**: Restrict database access to ML team only
4. **Encryption**: Data at rest and in transit encrypted
5. **Compliance**: GDPR/CCPA compliant (data minimization)

---

## Contact

For questions or issues, contact:
- DWCP Team: dwcp-team@novacron.com
- ML Team: ml-team@novacron.com
