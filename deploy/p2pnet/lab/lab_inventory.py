#!/usr/bin/python3
"""Build the deterministic three-node test inventory from real guest public keys."""
import argparse
import pathlib
import sys

import yaml


def read_key(path: pathlib.Path, wireguard: bool = False) -> str:
    value = path.read_text(encoding="utf-8").strip().split()
    if not value or (not wireguard and len(value) < 2):
        raise ValueError(f"invalid public key file: {path}")
    return value[0] if wireguard else " ".join(value[:2])




def inventory(keys_dir: pathlib.Path) -> dict:
    nodes = []
    for index in range(1, 4):
        name = f"node{index}"
        prefix = keys_dir / name
        node = {
            "name": name,
            "overlay_ipv4": f"10.77.0.{index}",
            "overlay_ipv6": None,
            "wg_pubkey": read_key(prefix / "wg0.pub", wireguard=True),
            "endpoint": f"192.0.2.1{index}:51820",
            "underlay_family": "ipv4",
            "wan_mtu": 1500,
            "uplink_mbit": 200,
            "downlink_mbit": 200,
            "wan": {"interface": "wan1", "gateway": None},
            "wan2": None,
            "ssh_pubkey": read_key(prefix / "ssh.pub"),
            "ssh_bulk_pubkey": read_key(prefix / "bulk.pub"),
            "ssh_hostkey": read_key(prefix / "host.pub"),
            "underlay_iperf": False,
            "zfs": {"replica_root": "tank/p2p-replicas"},
            "images_dir": "/var/lib/p2pnet/images",
            "capacity": {"vcpus": 4, "ram_mib": 4096},
        }
        if index == 1:
            node["wan2"] = {
                "interface": "wan2",
                "gateway": None,
                "mtu": 1500,
                "endpoint": "192.0.2.21:51821",
                "wg1_pubkey": read_key(prefix / "wg1.pub", wireguard=True),
                "overlay_ipv4": "10.77.1.1",
                "uplink_mbit": 200,
            }
        nodes.append(node)
    return {
        "version": 1,
        "cluster": {
            "name": "p2pnet-lab",
            "overlay": {
                "ipv4_cidr": "10.77.0.0/24",
                "ipv6_cidr": None,
                "plane2_ipv4_cidr": "10.77.1.0/24",
                "wg_port": 51820,
                "wg1_port": 51821,
            },
            "vxlan": {"vni": 7700, "port": 4789, "bridge": "br-p2p"},
            "qos": {
                "shape_percent": 94,
                "night_start": "02:00",
                "night_end": "06:00",
                "bulk_day_ceil_percent": 10,
            },
            "migration": {"port_min": 49152, "port_max": 49215},
            "replication_defaults": {"interval_min": 15, "keep_last": 96, "keep_daily": 14},
            "alert_webhook": None,
        },
        "nodes": nodes,
        "replication": [
            {
                "name": "labjob",
                "source": "node1",
                "dataset": "tank/vms",
                "targets": ["node2", "node3"],
                "interval_min": 5,
                "keep_last": 3,
                "keep_daily": 1,
            }
        ],
        "swarm": {"base_images": ["lab-base.img"]},
        "vms": [
            {
                "name": "lab-vm",
                "disk_gib": 1,
                "vcpus": 1,
                "ram_mib": 256,
                "replicas": ["node1", "node2"],
                "running_on": "node1",
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        data = inventory(args.keys_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"lab_inventory: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
