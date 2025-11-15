#!/usr/bin/env python3
"""
Generate synthetic DWCP training data with realistic network patterns
for bandwidth predictor training
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_realistic_metrics(num_samples=10000, seed=42):
    """Generate realistic network metrics with temporal patterns"""
    np.random.seed(seed)

    # Generate timestamps (1 sample per minute for ~7 days)
    start_time = datetime.now() - timedelta(minutes=num_samples)
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_samples)]

    # Extract temporal features
    hours = np.array([t.hour for t in timestamps])
    days = np.array([t.weekday() for t in timestamps])

    # Base patterns with daily/weekly cycles
    # Business hours (9-17) have higher bandwidth
    business_hours_factor = np.where((hours >= 9) & (hours <= 17), 1.5, 1.0)
    # Weekends have lower usage
    weekend_factor = np.where(days >= 5, 0.7, 1.0)

    # Generate correlated metrics

    # 1. Throughput (10-1000 Mbps with patterns)
    base_throughput = 500 + 200 * np.sin(hours * 2 * np.pi / 24)  # Daily cycle
    throughput_mbps = base_throughput * business_hours_factor * weekend_factor
    throughput_mbps += np.random.normal(0, 50, num_samples)  # Noise
    throughput_mbps = np.clip(throughput_mbps, 10, 1000)

    # 2. RTT (inversely correlated with throughput)
    base_rtt = 50 - (throughput_mbps - 500) / 20  # Higher bandwidth = lower RTT
    rtt_ms = base_rtt + np.random.normal(0, 5, num_samples)
    rtt_ms = np.clip(rtt_ms, 5, 150)

    # 3. Jitter (correlated with RTT variance)
    jitter_ms = rtt_ms * 0.1 + np.random.exponential(2, num_samples)
    jitter_ms = np.clip(jitter_ms, 0.1, 20)

    # 4. Packet loss (higher during congestion)
    congestion_score = (1000 - throughput_mbps) / 1000  # 0-1
    packet_loss = congestion_score * 0.05 + np.random.exponential(0.001, num_samples)
    packet_loss = np.clip(packet_loss, 0, 0.1)

    # 5. Network tiers (Premium, Standard, Economy)
    network_tier = np.random.choice(['Premium', 'Standard', 'Economy'], num_samples, p=[0.3, 0.5, 0.2])

    # 6. Link types
    link_type = np.random.choice(['dc', 'metro', 'wan'], num_samples, p=[0.4, 0.4, 0.2])

    # 7. Regions and AZs
    regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
    region = np.random.choice(regions, num_samples)
    az = np.array([f"{r}{np.random.choice(['a', 'b', 'c'])}" for r in region])

    # 8. Bytes transferred (correlated with throughput)
    bytes_tx = (throughput_mbps * 1024 * 1024 / 8) * 60 * np.random.uniform(0.8, 1.2, num_samples)
    bytes_rx = bytes_tx * np.random.uniform(0.9, 1.1, num_samples)

    # 9. Retransmits (correlated with packet loss)
    retransmits = (packet_loss * 1000 + np.random.poisson(5, num_samples)).astype(int)

    # 10. Congestion window (correlated with throughput)
    congestion_window = (throughput_mbps * 2 + np.random.normal(0, 100, num_samples)).astype(int)
    congestion_window = np.clip(congestion_window, 100, 2000)

    # 11. Queue depth (correlated with congestion)
    queue_depth = (congestion_score * 100 + np.random.exponential(10, num_samples)).astype(int)
    queue_depth = np.clip(queue_depth, 0, 200)

    # 12. DWCP mode
    dwcp_mode = np.random.choice(['standard', 'turbo', 'reliable'], num_samples, p=[0.5, 0.3, 0.2])

    # 13. Transport type
    transport_type = np.random.choice(['tcp', 'quic', 'udp'], num_samples, p=[0.6, 0.3, 0.1])

    # 14. Node and peer IDs
    node_ids = np.random.randint(1000, 2000, num_samples)
    peer_ids = np.random.randint(2000, 3000, num_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': [int(t.timestamp()) for t in timestamps],
        'region': region,
        'az': az,
        'link_type': link_type,
        'node_id': node_ids,
        'peer_id': peer_ids,
        'rtt_ms': rtt_ms,
        'jitter_ms': jitter_ms,
        'throughput_mbps': throughput_mbps,
        'bytes_tx': bytes_tx.astype(int),
        'bytes_rx': bytes_rx.astype(int),
        'packet_loss': packet_loss,
        'retransmits': retransmits,
        'congestion_window': congestion_window,
        'queue_depth': queue_depth,
        'dwcp_mode': dwcp_mode,
        'network_tier': network_tier,
        'transport_type': transport_type,
        'time_of_day': hours,
        'day_of_week': days,
        # Legacy column names for compatibility
        'bandwidth_mbps': throughput_mbps,
        'latency_ms': rtt_ms,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate DWCP training data')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    print(f"Generating {args.samples} training samples...")
    df = generate_realistic_metrics(args.samples, args.seed)

    print(f"Saving to {args.output}...")
    df.to_csv(args.output, index=False)

    print("\nData Statistics:")
    print(df.describe())
    print(f"\nTotal samples: {len(df)}")
    print(f"Date range: {datetime.fromtimestamp(df['timestamp'].min())} to {datetime.fromtimestamp(df['timestamp'].max())}")
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
