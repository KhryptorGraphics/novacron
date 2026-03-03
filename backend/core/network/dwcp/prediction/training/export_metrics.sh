#!/bin/bash
# Export network metrics from Prometheus for model training

set -e

PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/training_data.csv}"
DAYS_BACK="${DAYS_BACK:-30}"

echo "Exporting network metrics from Prometheus"
echo "URL: $PROMETHEUS_URL"
echo "Output: $OUTPUT_FILE"
echo "Days back: $DAYS_BACK"

# Calculate time range
END_TIME=$(date +%s)
START_TIME=$((END_TIME - (DAYS_BACK * 86400)))

echo "Time range: $(date -d @$START_TIME) to $(date -d @$END_TIME)"

# Export metrics using Prometheus API
QUERY='
{
  "query": "dwcp_pba_current_bandwidth_mbps",
  "start": "'$START_TIME'",
  "end": "'$END_TIME'",
  "step": "60"
}
'

# Fetch bandwidth metrics
echo "Fetching bandwidth metrics..."
BANDWIDTH_DATA=$(curl -s -G "$PROMETHEUS_URL/api/v1/query_range" \
  --data-urlencode "query=dwcp_pba_current_bandwidth_mbps" \
  --data-urlencode "start=$START_TIME" \
  --data-urlencode "end=$END_TIME" \
  --data-urlencode "step=60")

# Fetch latency metrics
echo "Fetching latency metrics..."
LATENCY_DATA=$(curl -s -G "$PROMETHEUS_URL/api/v1/query_range" \
  --data-urlencode "query=dwcp_pba_current_latency_ms" \
  --data-urlencode "start=$START_TIME" \
  --data-urlencode "end=$END_TIME" \
  --data-urlencode "step=60")

# Fetch packet loss metrics
echo "Fetching packet loss metrics..."
PACKET_LOSS_DATA=$(curl -s -G "$PROMETHEUS_URL/api/v1/query_range" \
  --data-urlencode "query=dwcp_pba_current_packet_loss_ratio" \
  --data-urlencode "start=$START_TIME" \
  --data-urlencode "end=$END_TIME" \
  --data-urlencode "step=60")

# Fetch jitter metrics
echo "Fetching jitter metrics..."
JITTER_DATA=$(curl -s -G "$PROMETHEUS_URL/api/v1/query_range" \
  --data-urlencode "query=dwcp_pba_current_jitter_ms" \
  --data-urlencode "start=$START_TIME" \
  --data-urlencode "end=$END_TIME" \
  --data-urlencode "step=60")

# Process and combine data using Python
python3 <<EOF
import json
import csv
from datetime import datetime

# Load JSON data
bandwidth = json.loads('''$BANDWIDTH_DATA''')
latency = json.loads('''$LATENCY_DATA''')
packet_loss = json.loads('''$PACKET_LOSS_DATA''')
jitter = json.loads('''$JITTER_DATA''')

# Extract time series
def extract_timeseries(data):
    if data['status'] == 'success' and data['data']['result']:
        return {int(ts): float(val) for ts, val in data['data']['result'][0]['values']}
    return {}

bandwidth_ts = extract_timeseries(bandwidth)
latency_ts = extract_timeseries(latency)
packet_loss_ts = extract_timeseries(packet_loss)
jitter_ts = extract_timeseries(jitter)

# Get all timestamps
all_timestamps = set(bandwidth_ts.keys()) & set(latency_ts.keys()) & \
                 set(packet_loss_ts.keys()) & set(jitter_ts.keys())

# Write CSV
with open('$OUTPUT_FILE', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp',
        'bandwidth_mbps',
        'latency_ms',
        'packet_loss',
        'jitter_ms',
        'time_of_day',
        'day_of_week'
    ])

    for ts in sorted(all_timestamps):
        dt = datetime.fromtimestamp(ts)
        writer.writerow([
            ts,
            bandwidth_ts.get(ts, 0),
            latency_ts.get(ts, 0),
            packet_loss_ts.get(ts, 0),
            jitter_ts.get(ts, 0),
            dt.hour,
            dt.weekday()
        ])

print(f"Exported {len(all_timestamps)} samples to $OUTPUT_FILE")
EOF

# Verify output
if [ -f "$OUTPUT_FILE" ]; then
    SAMPLE_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo "Export complete: $SAMPLE_COUNT samples"
    echo "First few lines:"
    head -5 "$OUTPUT_FILE"
else
    echo "Error: Failed to create output file"
    exit 1
fi
