#!/usr/bin/env python3
"""
Generate synthetic compression training data for DWCP Compression Selector

This script creates realistic synthetic data based on real-world compression
characteristics for initial model development and testing.
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_network_characteristics(n_samples: int, link_type: str) -> dict:
    """Generate network characteristics based on link type"""

    if link_type == 'dc':  # Datacenter
        rtt = np.random.gamma(2, 0.3, n_samples)  # 0.1-2ms typical
        jitter = rtt * np.random.uniform(0.05, 0.15, n_samples)
        bandwidth = np.random.uniform(1000, 10000, n_samples)  # 1-10 Gbps
        packet_loss = np.random.exponential(0.0001, n_samples)  # Very low

    elif link_type == 'metro':  # Metro
        rtt = np.random.gamma(3, 2.5, n_samples)  # 1-15ms typical
        jitter = rtt * np.random.uniform(0.1, 0.25, n_samples)
        bandwidth = np.random.uniform(100, 1000, n_samples)  # 100Mbps-1Gbps
        packet_loss = np.random.exponential(0.0005, n_samples)

    else:  # WAN
        rtt = np.random.gamma(5, 8, n_samples)  # 10-80ms typical
        jitter = rtt * np.random.uniform(0.15, 0.35, n_samples)
        bandwidth = np.random.uniform(10, 200, n_samples)  # 10-200Mbps
        packet_loss = np.random.exponential(0.002, n_samples)

    return {
        'rtt_ms': rtt,
        'jitter_ms': jitter,
        'available_bandwidth_mbps': bandwidth,
        'packet_loss_rate': np.clip(packet_loss, 0, 0.05)  # Cap at 5%
    }


def generate_data_characteristics(n_samples: int, data_type: str) -> dict:
    """Generate data characteristics based on data type"""

    if data_type == 'vm_memory':
        # Memory has patterns (page tables, stack frames)
        size = np.random.lognormal(15, 2, n_samples)  # ~1MB-100MB
        entropy = np.random.beta(2, 3, n_samples)  # Lower entropy (more compressible)
        compressibility = 1 - entropy * 0.8  # Inverse of entropy

    elif data_type == 'vm_disk':
        # Disk data is mixed (OS, apps, user data)
        size = np.random.lognormal(18, 2.5, n_samples)  # ~10MB-1GB
        entropy = np.random.beta(3, 2, n_samples)  # Higher entropy
        compressibility = 1 - entropy * 0.9

    else:  # database
        # Database has structured data
        size = np.random.lognormal(16, 1.5, n_samples)  # ~1MB-50MB
        entropy = np.random.beta(2.5, 2.5, n_samples)  # Moderate entropy
        compressibility = 1 - entropy * 0.85

    return {
        'data_size_bytes': size.astype(int),
        'entropy': entropy,
        'compressibility_score': compressibility
    }


def generate_hde_metrics(
    data_size: np.ndarray,
    compressibility: np.ndarray,
    cpu_usage: np.ndarray
) -> dict:
    """Generate HDE compression metrics"""

    # HDE compression ratio depends on data compressibility and delta hit rate
    base_ratio = 1 + compressibility * np.random.uniform(5, 40, len(data_size))
    delta_hit_rate = compressibility * np.random.uniform(60, 95, len(data_size))

    # Compression time depends on data size and CPU availability
    compression_time = (data_size / 1e6) * np.random.uniform(10, 30, len(data_size))
    compression_time *= (1 + cpu_usage)  # Slower when CPU is busy

    compressed_size = (data_size / base_ratio).astype(int)

    # Throughput
    throughput = (data_size / 1e6 * 8) / (compression_time / 1000)  # Mbps

    return {
        'hde_compression_ratio': base_ratio,
        'hde_delta_hit_rate': delta_hit_rate,
        'hde_compression_time_ms': compression_time,
        'hde_compressed_size_bytes': compressed_size,
        'hde_throughput_mbps': throughput
    }


def generate_amst_metrics(
    data_size: np.ndarray,
    bandwidth: np.ndarray,
    rtt: np.ndarray
) -> dict:
    """Generate AMST (multi-stream) metrics"""

    # AMST achieves good bandwidth utilization
    streams = np.random.choice([2, 4, 8], len(data_size))
    transfer_rate = bandwidth * np.random.uniform(0.7, 0.95, len(data_size))

    # Compression time is faster (simpler algorithm)
    compression_time = (data_size / 1e6) * np.random.uniform(5, 15, len(data_size))

    # Lower compression ratio (optimized for speed)
    compression_ratio = np.random.uniform(1.5, 5, len(data_size))
    compressed_size = (data_size / compression_ratio).astype(int)

    return {
        'amst_streams': streams,
        'amst_transfer_rate_mbps': transfer_rate,
        'amst_compression_time_ms': compression_time,
        'amst_compressed_size_bytes': compressed_size
    }


def generate_baseline_metrics(
    data_size: np.ndarray,
    compressibility: np.ndarray
) -> dict:
    """Generate baseline (standard Zstd) metrics"""

    # Standard compression ratio
    compression_ratio = 1 + compressibility * np.random.uniform(2, 8, len(data_size))

    # Compression time
    compression_time = (data_size / 1e6) * np.random.uniform(3, 10, len(data_size))

    compressed_size = (data_size / compression_ratio).astype(int)

    return {
        'baseline_compression_ratio': compression_ratio,
        'baseline_compression_time_ms': compression_time,
        'baseline_compressed_size_bytes': compressed_size
    }


def generate_synthetic_data(
    n_samples: int = 100000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate comprehensive synthetic compression training data

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with all required features
    """
    np.random.seed(seed)

    print(f"Generating {n_samples:,} synthetic samples...")

    # Distribution across link types and data types
    link_types = np.random.choice(
        ['dc', 'metro', 'wan'],
        n_samples,
        p=[0.4, 0.3, 0.3]
    )
    data_types = np.random.choice(
        ['vm_memory', 'vm_disk', 'database'],
        n_samples,
        p=[0.5, 0.3, 0.2]
    )

    # Base metadata
    data = {
        'timestamp': [
            datetime.now() - timedelta(hours=i % (24*30))  # 30 days
            for i in range(n_samples)
        ],
        'region': np.random.choice(
            ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-south-1'],
            n_samples
        ),
        'az': np.random.choice(['a', 'b', 'c'], n_samples),
        'link_type': link_types,
        'node_id': [f'node-{np.random.randint(1000)}' for _ in range(n_samples)],
        'peer_id': [f'peer-{np.random.randint(1000)}' for _ in range(n_samples)],
        'data_type': data_types,
    }

    # Generate per link type
    for lt in ['dc', 'metro', 'wan']:
        mask = link_types == lt
        n = mask.sum()
        if n > 0:
            net_chars = generate_network_characteristics(n, lt)
            for key, values in net_chars.items():
                if key not in data:
                    data[key] = np.zeros(n_samples)
                data[key][mask] = values

    # Generate per data type
    for dt in ['vm_memory', 'vm_disk', 'database']:
        mask = data_types == dt
        n = mask.sum()
        if n > 0:
            data_chars = generate_data_characteristics(n, dt)
            for key, values in data_chars.items():
                if key not in data:
                    data[key] = np.zeros(n_samples)
                data[key][mask] = values

    # System state
    data['cpu_usage'] = np.random.beta(2, 5, n_samples)  # Tends toward low usage
    data['memory_available_mb'] = np.random.uniform(2048, 16384, n_samples).astype(int)
    data['active_transfers'] = np.random.poisson(3, n_samples)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # HDE metrics
    hde_metrics = generate_hde_metrics(
        df['data_size_bytes'].values,
        df['compressibility_score'].values,
        df['cpu_usage'].values
    )
    for key, values in hde_metrics.items():
        df[key] = values

    # AMST metrics
    amst_metrics = generate_amst_metrics(
        df['data_size_bytes'].values,
        df['available_bandwidth_mbps'].values,
        df['rtt_ms'].values
    )
    for key, values in amst_metrics.items():
        df[key] = values

    # Baseline metrics
    baseline_metrics = generate_baseline_metrics(
        df['data_size_bytes'].values,
        df['compressibility_score'].values
    )
    for key, values in baseline_metrics.items():
        df[key] = values

    # Current compression (simulated selection)
    # This will be replaced by oracle computation during training
    current_compression = []
    for _, row in df.iterrows():
        if row['link_type'] == 'dc' and row['available_bandwidth_mbps'] > 500:
            current_compression.append('amst')
        elif row['hde_compression_ratio'] > 15 and row['hde_delta_hit_rate'] > 80:
            current_compression.append('hde')
        elif row['available_bandwidth_mbps'] < 100:
            current_compression.append('hde')
        else:
            current_compression.append('baseline')

    df['current_compression'] = current_compression

    # Actual performance metrics (for oracle computation)
    df['actual_transfer_time_ms'] = 0.0  # Placeholder
    df['actual_throughput_mbps'] = 0.0  # Placeholder
    df['total_cpu_overhead_ms'] = 0.0  # Placeholder

    for idx, row in df.iterrows():
        algo = row['current_compression']

        if algo == 'hde':
            transfer_time = (row['hde_compressed_size_bytes'] / 1e6 * 8) / row['available_bandwidth_mbps'] * 1000
            transfer_time += row['rtt_ms'] * 2
            cpu_overhead = row['hde_compression_time_ms']

        elif algo == 'amst':
            transfer_time = (row['amst_compressed_size_bytes'] / 1e6 * 8) / row['amst_transfer_rate_mbps'] * 1000
            transfer_time += row['rtt_ms'] * 2
            cpu_overhead = row['amst_compression_time_ms']

        else:  # baseline
            transfer_time = (row['baseline_compressed_size_bytes'] / 1e6 * 8) / row['available_bandwidth_mbps'] * 1000
            transfer_time += row['rtt_ms'] * 2
            cpu_overhead = row['baseline_compression_time_ms']

        df.at[idx, 'actual_transfer_time_ms'] = transfer_time
        df.at[idx, 'actual_throughput_mbps'] = (row['data_size_bytes'] / 1e6 * 8) / (transfer_time / 1000)
        df.at[idx, 'total_cpu_overhead_ms'] = cpu_overhead

    print(f"Generated {len(df):,} samples")
    print(f"\nLink type distribution:")
    print(df['link_type'].value_counts())
    print(f"\nData type distribution:")
    print(df['data_type'].value_counts())
    print(f"\nCurrent compression distribution:")
    print(df['current_compression'].value_counts())

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic compression training data'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100000,
        help='Number of samples to generate (default: 100000)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    # Generate data
    df = generate_synthetic_data(n_samples=args.samples, seed=args.seed)

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df):,} samples to {args.output}")

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total size: {df['data_size_bytes'].sum() / 1e9:.2f} GB")
    print(f"  Avg RTT: {df['rtt_ms'].mean():.2f} ms")
    print(f"  Avg bandwidth: {df['available_bandwidth_mbps'].mean():.2f} Mbps")
    print(f"  Avg HDE compression ratio: {df['hde_compression_ratio'].mean():.2f}x")
    print(f"  Avg AMST transfer rate: {df['amst_transfer_rate_mbps'].mean():.2f} Mbps")


if __name__ == '__main__':
    main()
