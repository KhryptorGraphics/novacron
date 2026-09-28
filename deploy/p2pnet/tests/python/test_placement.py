#!/usr/bin/python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libexec"))
import inventory


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "nodes": [
                {"name": "node-a", "capacity": {"vcpus": 8, "ram_mib": 8192},
                 "uplink_mbit": 100},
                {"name": "node-b", "capacity": {"vcpus": 8, "ram_mib": 8192},
                 "uplink_mbit": 500},
                {"name": "node-c", "capacity": {"vcpus": 8, "ram_mib": 16384},
                 "uplink_mbit": 100},
            ],
            "vms": [
                {"name": "web01", "disk_gib": 40, "vcpus": 2, "ram_mib": 2048,
                 "replicas": ["node-a", "node-b"], "running_on": "node-a"},
                {"name": "other", "disk_gib": 1, "vcpus": 4, "ram_mib": 4096,
                 "replicas": ["node-a"], "running_on": "node-a"},
            ],
        }

    def test_stays_on_existing_replica_when_it_fits(self):
        result = inventory.placement(self.data, "web01")
        self.assertEqual((result["node"], result["action"], result["wan_copy_gib"]),
                         ("node-a", "stay", 0))

    def test_moves_to_first_replica_with_capacity(self):
        vm = self.data["vms"][0]
        vm["running_on"] = "node-c"
        vm["replicas"] = ["node-b", "node-a"]
        result = inventory.placement(self.data, "web01")
        self.assertEqual((result["node"], result["action"], result["wan_copy_gib"]),
                         ("node-b", "move", 0))

    def test_full_copy_chooses_most_free_ram_then_uplink(self):
        vm = self.data["vms"][0]
        vm["running_on"] = None
        vm["replicas"] = []
        result = inventory.placement(self.data, "web01")
        self.assertEqual((result["node"], result["action"], result["wan_copy_gib"]),
                         ("node-c", "copy", 40))

    def test_capacity_exhaustion_is_rejected(self):
        for node in self.data["nodes"]:
            node["capacity"] = {"vcpus": 1, "ram_mib": 512}
        with self.assertRaisesRegex(ValueError, "no node has capacity for web01"):
            inventory.placement(self.data, "web01")


if __name__ == "__main__":
    unittest.main()
