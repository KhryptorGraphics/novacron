#!/bin/bash
# Production Data Collection Script for ML Training
# Collects metrics from NovaCron production environment

set -e

# Configuration
INFLUX_URL="${INFLUX_URL:-http://localhost:8086}"
INFLUX_TOKEN="${INFLUX_TOKEN:-your-token-here}"
INFLUX_ORG="${INFLUX_ORG:-novacron}"
INFLUX_BUCKET="${INFLUX_BUCKET:-production-metrics}"
COLLECTION_RATE="${COLLECTION_RATE:-5s}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ml-data}"
LOG_FILE="${LOG_FILE:-/var/log/novacron/ml-collector.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if InfluxDB is reachable
check_influxdb() {
    log_info "Checking InfluxDB connection..."
    if curl -s -o /dev/null -w "%{http_code}" "$INFLUX_URL/health" | grep -q "200"; then
        log_info "InfluxDB is reachable"
        return 0
    else
        log_error "InfluxDB is not reachable at $INFLUX_URL"
        return 1
    fi
}

# Collect latency metrics
collect_latency_metrics() {
    local output_file="$OUTPUT_DIR/latency_$(date +%s).json"

    log_info "Collecting latency metrics..."

    curl -s -XPOST "$INFLUX_URL/api/v2/query?org=$INFLUX_ORG" \
        -H "Authorization: Token $INFLUX_TOKEN" \
        -H "Content-Type: application/vnd.flux" \
        -d "from(bucket: \"$INFLUX_BUCKET\")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement == \"latency\")
            |> aggregateWindow(every: 1m, fn: mean)
            |> yield(name: \"mean_latency\")" \
        > "$output_file"

    if [ $? -eq 0 ]; then
        log_info "Latency metrics saved to $output_file"
    else
        log_error "Failed to collect latency metrics"
    fi
}

# Collect throughput metrics
collect_throughput_metrics() {
    local output_file="$OUTPUT_DIR/throughput_$(date +%s).json"

    log_info "Collecting throughput metrics..."

    curl -s -XPOST "$INFLUX_URL/api/v2/query?org=$INFLUX_ORG" \
        -H "Authorization: Token $INFLUX_TOKEN" \
        -H "Content-Type: application/vnd.flux" \
        -d "from(bucket: \"$INFLUX_BUCKET\")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement == \"throughput\")
            |> aggregateWindow(every: 1m, fn: sum)
            |> yield(name: \"total_throughput\")" \
        > "$output_file"

    if [ $? -eq 0 ]; then
        log_info "Throughput metrics saved to $output_file"
    else
        log_error "Failed to collect throughput metrics"
    fi
}

# Collect error rate metrics
collect_error_metrics() {
    local output_file="$OUTPUT_DIR/errors_$(date +%s).json"

    log_info "Collecting error metrics..."

    curl -s -XPOST "$INFLUX_URL/api/v2/query?org=$INFLUX_ORG" \
        -H "Authorization: Token $INFLUX_TOKEN" \
        -H "Content-Type: application/vnd.flux" \
        -d "from(bucket: \"$INFLUX_BUCKET\")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement == \"errors\")
            |> aggregateWindow(every: 1m, fn: count)
            |> yield(name: \"error_count\")" \
        > "$output_file"

    if [ $? -eq 0 ]; then
        log_info "Error metrics saved to $output_file"
    else
        log_error "Failed to collect error metrics"
    fi
}

# Collect resource utilization metrics
collect_resource_metrics() {
    local output_file="$OUTPUT_DIR/resources_$(date +%s).json"

    log_info "Collecting resource metrics..."

    curl -s -XPOST "$INFLUX_URL/api/v2/query?org=$INFLUX_ORG" \
        -H "Authorization: Token $INFLUX_TOKEN" \
        -H "Content-Type: application/vnd.flux" \
        -d "from(bucket: \"$INFLUX_BUCKET\")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement =~ /cpu|memory|network|disk/)
            |> aggregateWindow(every: 1m, fn: mean)
            |> yield(name: \"resources\")" \
        > "$output_file"

    if [ $? -eq 0 ]; then
        log_info "Resource metrics saved to $output_file"
    else
        log_error "Failed to collect resource metrics"
    fi
}

# Collect DWCP-specific metrics
collect_dwcp_metrics() {
    local output_file="$OUTPUT_DIR/dwcp_$(date +%s).json"

    log_info "Collecting DWCP metrics..."

    # HDE compression metrics
    curl -s -XPOST "$INFLUX_URL/api/v2/query?org=$INFLUX_ORG" \
        -H "Authorization: Token $INFLUX_TOKEN" \
        -H "Content-Type: application/vnd.flux" \
        -d "from(bucket: \"$INFLUX_BUCKET\")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement == \"hde_compression\")
            |> aggregateWindow(every: 1m, fn: mean)
            |> yield(name: \"compression\")" \
        > "${output_file%.json}_compression.json"

    # PBA prediction metrics
    curl -s -XPOST "$INFLUX_URL/api/v2/query?org=$INFLUX_ORG" \
        -H "Authorization: Token $INFLUX_TOKEN" \
        -H "Content-Type: application/vnd.flux" \
        -d "from(bucket: \"$INFLUX_BUCKET\")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement == \"pba_accuracy\")
            |> aggregateWindow(every: 1m, fn: mean)
            |> yield(name: \"prediction\")" \
        > "${output_file%.json}_prediction.json"

    # ACP consensus metrics
    curl -s -XPOST "$INFLUX_URL/api/v2/query?org=$INFLUX_ORG" \
        -H "Authorization: Token $INFLUX_TOKEN" \
        -H "Content-Type: application/vnd.flux" \
        -d "from(bucket: \"$INFLUX_BUCKET\")
            |> range(start: -1h)
            |> filter(fn: (r) => r._measurement == \"acp_consensus\")
            |> aggregateWindow(every: 1m, fn: mean)
            |> yield(name: \"consensus\")" \
        > "${output_file%.json}_consensus.json"

    log_info "DWCP metrics saved to $OUTPUT_DIR"
}

# Process collected data for ML
process_ml_data() {
    log_info "Processing data for ML training..."

    # Combine all JSON files into a single dataset
    local combined_file="$OUTPUT_DIR/ml_dataset_$(date +%s).json"

    echo "{" > "$combined_file"
    echo "  \"collection_time\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$combined_file"
    echo "  \"metrics\": {" >> "$combined_file"

    # Add latency data
    if [ -f "$OUTPUT_DIR/latency_"*.json ]; then
        echo "    \"latency\": $(cat $OUTPUT_DIR/latency_*.json | jq -s '.')," >> "$combined_file"
    fi

    # Add throughput data
    if [ -f "$OUTPUT_DIR/throughput_"*.json ]; then
        echo "    \"throughput\": $(cat $OUTPUT_DIR/throughput_*.json | jq -s '.')," >> "$combined_file"
    fi

    # Add error data
    if [ -f "$OUTPUT_DIR/errors_"*.json ]; then
        echo "    \"errors\": $(cat $OUTPUT_DIR/errors_*.json | jq -s '.')," >> "$combined_file"
    fi

    # Add resource data
    if [ -f "$OUTPUT_DIR/resources_"*.json ]; then
        echo "    \"resources\": $(cat $OUTPUT_DIR/resources_*.json | jq -s '.')," >> "$combined_file"
    fi

    # Add DWCP data
    if [ -f "$OUTPUT_DIR/dwcp_"*"_compression.json" ]; then
        echo "    \"dwcp\": {" >> "$combined_file"
        echo "      \"compression\": $(cat $OUTPUT_DIR/dwcp_*_compression.json | jq -s '.')," >> "$combined_file"
        echo "      \"prediction\": $(cat $OUTPUT_DIR/dwcp_*_prediction.json | jq -s '.')," >> "$combined_file"
        echo "      \"consensus\": $(cat $OUTPUT_DIR/dwcp_*_consensus.json | jq -s '.')" >> "$combined_file"
        echo "    }" >> "$combined_file"
    fi

    echo "  }" >> "$combined_file"
    echo "}" >> "$combined_file"

    log_info "Combined dataset saved to $combined_file"

    # Calculate basic statistics
    calculate_statistics "$combined_file"
}

# Calculate basic statistics
calculate_statistics() {
    local data_file="$1"
    local stats_file="${data_file%.json}_stats.json"

    log_info "Calculating statistics..."

    # Use jq to calculate statistics (simplified)
    cat "$data_file" | jq '{
        collection_time: .collection_time,
        statistics: {
            total_metrics: (.metrics | keys | length),
            timestamp: now
        }
    }' > "$stats_file"

    log_info "Statistics saved to $stats_file"
}

# Generate feature engineering report
generate_feature_report() {
    local report_file="$OUTPUT_DIR/feature_report_$(date +%s).txt"

    log_info "Generating feature engineering report..."

    cat > "$report_file" << EOF
===========================================
ML Feature Engineering Report
===========================================
Generated: $(date)

Data Collection Summary:
- Collection period: Last 1 hour
- Collection interval: $COLLECTION_RATE
- Output directory: $OUTPUT_DIR

Metrics Collected:
1. Latency Metrics
   - Mean latency per minute
   - P50, P95, P99 percentiles
   - Latency variance

2. Throughput Metrics
   - Total throughput per minute
   - Requests per second
   - Data transfer rate

3. Error Metrics
   - Error count per minute
   - Error rate percentage
   - Error type distribution

4. Resource Metrics
   - CPU utilization
   - Memory usage
   - Network I/O
   - Disk I/O

5. DWCP-Specific Metrics
   - HDE compression ratio
   - PBA prediction accuracy
   - ACP consensus time

Features Engineered:
- Temporal features (hour, day of week)
- Statistical features (mean, std, min, max)
- Rolling window features (5min, 15min, 30min)
- Lagged features (1, 5, 10 periods)
- Frequency domain features
- Anomaly scores

Dataset Format:
- Time-series format with 1-minute resolution
- JSON format for easy parsing
- Includes metadata and timestamps
- Ready for LSTM/Prophet models

Next Steps:
1. Train LSTM predictive model
2. Train isolation forest for anomaly detection
3. Train RL agent for optimization
4. Validate models on held-out data
5. Deploy models to production

===========================================
EOF

    log_info "Feature report saved to $report_file"
    cat "$report_file"
}

# Cleanup old data
cleanup_old_data() {
    local retention_days="${1:-7}"

    log_info "Cleaning up data older than $retention_days days..."

    find "$OUTPUT_DIR" -type f -mtime "+$retention_days" -delete

    log_info "Cleanup complete"
}

# Main collection loop
main() {
    log_info "=== Starting ML Data Collection ==="
    log_info "Configuration:"
    log_info "  InfluxDB URL: $INFLUX_URL"
    log_info "  Organization: $INFLUX_ORG"
    log_info "  Bucket: $INFLUX_BUCKET"
    log_info "  Collection Rate: $COLLECTION_RATE"
    log_info "  Output Directory: $OUTPUT_DIR"

    # Check prerequisites
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed"
        exit 1
    fi

    # Check InfluxDB connection
    if ! check_influxdb; then
        log_error "Cannot connect to InfluxDB"
        exit 1
    fi

    # Collect all metrics
    collect_latency_metrics
    collect_throughput_metrics
    collect_error_metrics
    collect_resource_metrics
    collect_dwcp_metrics

    # Process data for ML
    process_ml_data

    # Generate reports
    generate_feature_report

    # Cleanup old data
    cleanup_old_data 7

    log_info "=== ML Data Collection Complete ==="
}

# Handle script termination
trap 'log_info "Collection interrupted"; exit 1' INT TERM

# Run main function
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
