# DWCP Compression Data Collection Pipeline

## Overview

This document specifies the data collection, processing, and preparation pipeline for training the ML-based Compression Selector.

---

## 1. Data Collection Architecture

### 1.1 Production Data Sources

```
┌─────────────────────────────────────────────────────────────┐
│              DWCP Production Environment                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ HDE Engine │  │ AMST Engine│  │  Baseline  │           │
│  │            │  │            │  │  Engine    │           │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘           │
│        │               │               │                   │
│        └───────────────┴───────────────┘                   │
│                        │                                    │
│              ┌─────────▼──────────┐                        │
│              │ Metrics Collector  │                        │
│              │ (Go instrumentation)                        │
│              └─────────┬──────────┘                        │
│                        │                                    │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Time-Series DB     │
              │   (InfluxDB)         │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   ETL Pipeline       │
              │   (Python/Airflow)   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Training Dataset    │
              │  (CSV/Parquet)       │
              └──────────────────────┘
```

### 1.2 Instrumentation Points

Each compression operation logs:

**Go Code Instrumentation** (`backend/core/network/dwcp/compression/`):

```go
// In delta_encoder.go, adaptive_compression.go, baseline_sync.go
type CompressionMetric struct {
    // Metadata
    Timestamp       time.Time
    Region          string
    AvailabilityZone string
    NodeID          string
    PeerID          string

    // Network state
    LinkType        string  // dc, metro, wan
    RTTMs           float64
    JitterMs        float64
    BandwidthMbps   float64
    PacketLossRate  float64

    // Data characteristics
    DataSizeBytes   int64
    DataType        string  // vm_memory, vm_disk, database
    Entropy         float64
    Compressibility float64

    // System state
    CPUUsage        float64
    MemoryAvailableMb int64
    ActiveTransfers int

    // HDE metrics (if executed)
    HDECompressionRatio   float64
    HDEDeltaHitRate       float64
    HDECompressionTimeMs  float64
    HDECompressedSizeBytes int64
    HDEThroughputMbps     float64

    // AMST metrics (if executed)
    AMSTStreams           int
    AMSTTransferRateMbps  float64
    AMSTCompressionTimeMs float64
    AMSTCompressedSizeBytes int64

    // Baseline metrics (always executed)
    BaselineCompressionRatio float64
    BaselineCompressionTimeMs float64
    BaselineCompressedSizeBytes int64

    // Selected algorithm
    CurrentCompression string  // hde, amst, baseline

    // Performance outcomes
    ActualTransferTimeMs float64
    ActualThroughputMbps float64
    TotalCPUOverheadMs   float64
}
```

---

## 2. Data Schema

### 2.1 Training Data CSV Format

```csv
timestamp,region,az,link_type,node_id,peer_id,data_size_bytes,data_type,entropy,compressibility_score,rtt_ms,jitter_ms,available_bandwidth_mbps,packet_loss_rate,cpu_usage,memory_available_mb,active_transfers,hde_compression_ratio,hde_delta_hit_rate,hde_compression_time_ms,hde_compressed_size_bytes,hde_throughput_mbps,amst_streams,amst_transfer_rate_mbps,amst_compression_time_ms,amst_compressed_size_bytes,baseline_compression_ratio,baseline_compression_time_ms,baseline_compressed_size_bytes,current_compression,actual_transfer_time_ms,actual_throughput_mbps,total_cpu_overhead_ms
```

### 2.2 Example Data Row

```csv
2025-11-14T12:34:56Z,us-east-1,us-east-1a,metro,node-123,node-456,5242880,vm_memory,0.65,0.82,8.5,1.2,450.0,0.001,0.45,8192,3,15.2,85.5,12.3,344064,412.5,4,425.0,8.1,1048576,3.2,5.8,1638400,hde,125.4,398.2,18.1
```

### 2.3 Required Fields

**Minimum Required Fields** (for training):
- `timestamp`, `link_type`, `data_size_bytes`
- `rtt_ms`, `available_bandwidth_mbps`
- `cpu_usage`, `memory_available_mb`
- `hde_compression_ratio`, `hde_delta_hit_rate`
- `amst_transfer_rate_mbps`
- `baseline_compression_ratio`
- Performance metrics for oracle computation

**Optional Fields** (improve accuracy):
- `region`, `az` (for geo-specific models)
- `data_type` (for type-specific optimization)
- `entropy`, `compressibility_score` (for data profiling)

---

## 3. Data Collection Implementation

### 3.1 Go Instrumentation Code

**File**: `backend/core/network/dwcp/compression/metrics_collector.go`

```go
package compression

import (
    "context"
    "encoding/json"
    "time"

    "github.com/influxdata/influxdb-client-go/v2"
    "go.uber.org/zap"
)

// MetricsCollector collects compression metrics for ML training
type MetricsCollector struct {
    influxClient influxdb2.Client
    writeAPI     influxdb2.WriteAPIBlocking
    logger       *zap.Logger
    enabled      bool
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(influxURL, token, org, bucket string, logger *zap.Logger) *MetricsCollector {
    client := influxdb2.NewClient(influxURL, token)
    writeAPI := client.WriteAPIBlocking(org, bucket)

    return &MetricsCollector{
        influxClient: client,
        writeAPI:     writeAPI,
        logger:       logger,
        enabled:      true,
    }
}

// CollectCompressionMetrics logs detailed metrics for ML training
func (mc *MetricsCollector) CollectCompressionMetrics(
    ctx context.Context,
    metric *CompressionMetric,
) error {
    if !mc.enabled {
        return nil
    }

    // Create InfluxDB point
    point := influxdb2.NewPointWithMeasurement("compression_metrics").
        AddTag("region", metric.Region).
        AddTag("az", metric.AvailabilityZone).
        AddTag("link_type", metric.LinkType).
        AddTag("data_type", metric.DataType).
        AddTag("current_compression", metric.CurrentCompression).
        AddField("data_size_bytes", metric.DataSizeBytes).
        AddField("rtt_ms", metric.RTTMs).
        AddField("jitter_ms", metric.JitterMs).
        AddField("bandwidth_mbps", metric.BandwidthMbps).
        AddField("packet_loss_rate", metric.PacketLossRate).
        AddField("cpu_usage", metric.CPUUsage).
        AddField("memory_available_mb", metric.MemoryAvailableMb).
        AddField("entropy", metric.Entropy).
        AddField("compressibility", metric.Compressibility).
        AddField("hde_compression_ratio", metric.HDECompressionRatio).
        AddField("hde_delta_hit_rate", metric.HDEDeltaHitRate).
        AddField("hde_compression_time_ms", metric.HDECompressionTimeMs).
        AddField("hde_compressed_size_bytes", metric.HDECompressedSizeBytes).
        AddField("amst_streams", metric.AMSTStreams).
        AddField("amst_transfer_rate_mbps", metric.AMSTTransferRateMbps).
        AddField("amst_compression_time_ms", metric.AMSTCompressionTimeMs).
        AddField("amst_compressed_size_bytes", metric.AMSTCompressedSizeBytes).
        AddField("baseline_compression_ratio", metric.BaselineCompressionRatio).
        AddField("baseline_compression_time_ms", metric.BaselineCompressionTimeMs).
        AddField("baseline_compressed_size_bytes", metric.BaselineCompressedSizeBytes).
        AddField("actual_transfer_time_ms", metric.ActualTransferTimeMs).
        AddField("actual_throughput_mbps", metric.ActualThroughputMbps).
        AddField("total_cpu_overhead_ms", metric.TotalCPUOverheadMs).
        SetTime(metric.Timestamp)

    // Write to InfluxDB
    if err := mc.writeAPI.WritePoint(ctx, point); err != nil {
        mc.logger.Error("Failed to write compression metrics",
            zap.Error(err),
            zap.String("node_id", metric.NodeID))
        return err
    }

    return nil
}

// Close releases resources
func (mc *MetricsCollector) Close() {
    mc.influxClient.Close()
}
```

### 3.2 Integration with Compression Engines

**Modify `delta_encoder.go`**:

```go
func (de *DeltaEncoder) Encode(stateKey string, data []byte) (*EncodedData, error) {
    startTime := time.Now()

    // ... existing compression logic ...

    // Collect metrics for ML training
    if de.metricsCollector != nil {
        metric := &CompressionMetric{
            Timestamp:    time.Now(),
            DataSizeBytes: int64(len(data)),
            HDECompressionRatio: compressionRatio,
            HDEDeltaHitRate: deltaHitRate,
            HDECompressionTimeMs: float64(time.Since(startTime).Milliseconds()),
            // ... populate other fields ...
        }
        go de.metricsCollector.CollectCompressionMetrics(context.Background(), metric)
    }

    return encoded, nil
}
```

---

## 4. ETL Pipeline

### 4.1 InfluxDB to CSV Export

**Python Script**: `scripts/export_training_data.py`

```python
#!/usr/bin/env python3
"""Export compression metrics from InfluxDB to CSV for training"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
from influxdb_client import InfluxDBClient

def export_metrics(
    influx_url: str,
    token: str,
    org: str,
    bucket: str,
    start_time: datetime,
    end_time: datetime,
    output_file: str
):
    """Export metrics from InfluxDB to CSV"""

    client = InfluxDBClient(url=influx_url, token=token, org=org)
    query_api = client.query_api()

    # Flux query to extract all metrics
    query = f'''
    from(bucket: "{bucket}")
      |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
      |> filter(fn: (r) => r._measurement == "compression_metrics")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''

    # Execute query
    result = query_api.query_data_frame(query)

    # Convert to pandas DataFrame
    df = pd.DataFrame(result)

    # Rename columns and clean data
    df = df.rename(columns={'_time': 'timestamp'})

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} records to {output_file}")

    client.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--influx-url', required=True)
    parser.add_argument('--token', required=True)
    parser.add_argument('--org', required=True)
    parser.add_argument('--bucket', default='dwcp_metrics')
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.days)

    export_metrics(
        args.influx_url, args.token, args.org, args.bucket,
        start_time, end_time, args.output
    )
```

### 4.2 Airflow DAG for Automated Pipeline

**File**: `airflow/dags/compression_training_pipeline.py`

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 1),
    'email_on_failure': True,
    'email': ['ml-alerts@company.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'compression_selector_training',
    default_args=default_args,
    description='Weekly training of compression selector',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2am
    catchup=False
)

# Task 1: Export data from InfluxDB
export_task = PythonOperator(
    task_id='export_influxdb_data',
    python_callable=export_metrics_task,
    dag=dag
)

# Task 2: Train model
train_task = PythonOperator(
    task_id='train_compression_model',
    python_callable=train_model_task,
    dag=dag
)

# Task 3: Validate model
validate_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model_task,
    dag=dag
)

# Task 4: Deploy to staging
deploy_staging_task = PythonOperator(
    task_id='deploy_to_staging',
    python_callable=deploy_staging_task,
    dag=dag
)

# Define dependencies
export_task >> train_task >> validate_task >> deploy_staging_task
```

---

## 5. Data Quality Checks

### 5.1 Validation Rules

```python
def validate_training_data(df: pd.DataFrame) -> bool:
    """Validate data quality before training"""

    checks = []

    # 1. Required columns present
    required_cols = [
        'timestamp', 'data_size_bytes', 'rtt_ms', 'available_bandwidth_mbps',
        'hde_compression_ratio', 'amst_transfer_rate_mbps', 'baseline_compression_ratio'
    ]
    checks.append(all(col in df.columns for col in required_cols))

    # 2. No excessive missing values
    missing_pct = df.isnull().sum() / len(df)
    checks.append((missing_pct < 0.1).all())

    # 3. Reasonable value ranges
    checks.append((df['rtt_ms'] >= 0).all())
    checks.append((df['rtt_ms'] < 1000).all())  # < 1 second
    checks.append((df['available_bandwidth_mbps'] > 0).all())
    checks.append((df['cpu_usage'] >= 0).all() and (df['cpu_usage'] <= 1).all())

    # 4. Compression ratios are sensible
    checks.append((df['hde_compression_ratio'] >= 1).all())
    checks.append((df['hde_compression_ratio'] < 1000).all())

    # 5. Sufficient samples
    checks.append(len(df) >= 10000)

    # 6. Balanced classes (not too skewed)
    if 'current_compression' in df.columns:
        class_dist = df['current_compression'].value_counts(normalize=True)
        checks.append((class_dist > 0.05).all())  # Each class > 5%

    return all(checks)
```

### 5.2 Data Cleaning

```python
def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess training data"""

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove outliers (IQR method)
    for col in ['rtt_ms', 'available_bandwidth_mbps', 'data_size_bytes']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # Impute missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df
```

---

## 6. Sample Data Generation

### 6.1 Synthetic Data Generator

For initial development and testing:

**File**: `scripts/generate_synthetic_data.py`

```python
#!/usr/bin/env python3
"""Generate synthetic compression training data for testing"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples: int = 100000) -> pd.DataFrame:
    """Generate realistic synthetic compression metrics"""

    np.random.seed(42)

    # Network types with different characteristics
    link_types = np.random.choice(['dc', 'metro', 'wan'], n_samples, p=[0.4, 0.3, 0.3])

    data = {
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_samples)],
        'region': np.random.choice(['us-east-1', 'us-west-2', 'eu-west-1'], n_samples),
        'az': np.random.choice(['a', 'b', 'c'], n_samples),
        'link_type': link_types,
        'node_id': [f'node-{np.random.randint(1000)}' for _ in range(n_samples)],
        'peer_id': [f'peer-{np.random.randint(1000)}' for _ in range(n_samples)],
    }

    # Network characteristics based on link type
    data['rtt_ms'] = [
        np.random.gamma(2, 0.3) if lt == 'dc' else
        np.random.gamma(3, 2.5) if lt == 'metro' else
        np.random.gamma(5, 8)
        for lt in link_types
    ]
    data['jitter_ms'] = data['rtt_ms'] * np.random.uniform(0.1, 0.3, n_samples)
    data['available_bandwidth_mbps'] = [
        np.random.uniform(1000, 10000) if lt == 'dc' else
        np.random.uniform(100, 1000) if lt == 'metro' else
        np.random.uniform(10, 200)
        for lt in link_types
    ]
    data['packet_loss_rate'] = np.random.exponential(0.001, n_samples)

    # Data characteristics
    data['data_size_bytes'] = np.random.lognormal(15, 2, n_samples).astype(int)
    data['data_type'] = np.random.choice(['vm_memory', 'vm_disk', 'database'], n_samples)
    data['entropy'] = np.random.beta(3, 2, n_samples)
    data['compressibility_score'] = 1 - data['entropy']

    # System state
    data['cpu_usage'] = np.random.beta(2, 5, n_samples)
    data['memory_available_mb'] = np.random.uniform(2048, 16384, n_samples).astype(int)
    data['active_transfers'] = np.random.poisson(3, n_samples)

    # Compression metrics (HDE performs well on compressible data)
    data['hde_compression_ratio'] = 1 + data['compressibility_score'] * np.random.uniform(5, 40, n_samples)
    data['hde_delta_hit_rate'] = data['compressibility_score'] * np.random.uniform(60, 95, n_samples)
    data['hde_compression_time_ms'] = data['data_size_bytes'] / 1e6 * np.random.uniform(10, 30, n_samples)
    data['hde_compressed_size_bytes'] = (data['data_size_bytes'] / data['hde_compression_ratio']).astype(int)

    # AMST metrics
    data['amst_streams'] = np.random.choice([2, 4, 8], n_samples)
    data['amst_transfer_rate_mbps'] = data['available_bandwidth_mbps'] * np.random.uniform(0.7, 0.95, n_samples)
    data['amst_compression_time_ms'] = data['data_size_bytes'] / 1e6 * np.random.uniform(5, 15, n_samples)
    data['amst_compressed_size_bytes'] = (data['data_size_bytes'] / np.random.uniform(1.5, 5, n_samples)).astype(int)

    # Baseline metrics
    data['baseline_compression_ratio'] = np.random.uniform(2, 8, n_samples)
    data['baseline_compression_time_ms'] = data['data_size_bytes'] / 1e6 * np.random.uniform(3, 10, n_samples)
    data['baseline_compressed_size_bytes'] = (data['data_size_bytes'] / data['baseline_compression_ratio']).astype(int)

    # Current compression (for simulation)
    data['current_compression'] = np.random.choice(['hde', 'amst', 'baseline'], n_samples, p=[0.5, 0.3, 0.2])

    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    df = generate_synthetic_data(100000)
    df.to_csv('synthetic_compression_data.csv', index=False)
    print(f"Generated {len(df)} synthetic samples")
```

---

## 7. Deployment Checklist

- [ ] InfluxDB configured with `dwcp_metrics` bucket
- [ ] Metrics collector integrated in production Go code
- [ ] ETL pipeline deployed (Airflow DAG)
- [ ] Data validation checks in place
- [ ] Weekly training job scheduled
- [ ] Monitoring dashboards for data quality
- [ ] Alerting for pipeline failures

---

## 8. Monitoring & Observability

### 8.1 Key Metrics

**Data Collection Metrics**:
- Samples collected per hour
- Data volume (MB/day)
- Missing value percentage
- Class distribution (hde/amst/baseline)

**Pipeline Metrics**:
- ETL success rate
- Data export latency
- Training data freshness (hours since last update)

### 8.2 Grafana Dashboard

Monitor:
- Compression algorithm usage distribution
- Network latency trends (by region)
- Compression ratio trends
- CPU overhead trends

---

## Conclusion

This data pipeline ensures high-quality training data is continuously collected from production DWCP deployments, enabling weekly retraining and continuous improvement of the ML compression selector.
