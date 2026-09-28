#!/usr/bin/python3
import base64
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libexec"))
import inventory


SSH_A = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA node-a"
SSH_B = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA node-b"


def node(name, address, key_byte, endpoint):
    key = base64.b64encode(bytes((key_byte + i) % 256 for i in range(32))).decode()
    return {
        "name": name,
        "overlay_ipv4": address,
        "overlay_ipv6": None,
        "wg_pubkey": key,
        "endpoint": endpoint,
        "underlay_family": "ipv4",
        "wan_mtu": 1500,
        "uplink_mbit": 1000,
        "downlink_mbit": 1000,
        "wan": {"interface": "eth0", "gateway": None},
        "wan2": None,
        "ssh_pubkey": SSH_A if name == "node-a" else SSH_B,
        "ssh_bulk_pubkey": SSH_A if name == "node-a" else SSH_B,
        "ssh_hostkey": SSH_A if name == "node-a" else SSH_B,
        "zfs": {"replica_root": "tank/p2p-replicas"},
        "images_dir": "/var/lib/p2pnet/images",
        "capacity": {"vcpus": 16, "ram_mib": 65536},
    }


def base_inventory():
    return {
        "version": 1,
        "cluster": {
            "name": "test-cluster",
            "overlay": {"ipv4_cidr": "10.77.0.0/24", "ipv6_cidr": None,
                         "plane2_ipv4_cidr": "10.77.1.0/24", "wg_port": 51820, "wg1_port": 51821},
            "vxlan": {"vni": 7700, "port": 4789, "bridge": "br-p2p"},
            "qos": {"shape_percent": 94, "night_start": "02:00", "night_end": "06:00",
                    "bulk_day_ceil_percent": 10},
            "migration": {"port_min": 49152, "port_max": 49215},
            "replication_defaults": {"interval_min": 15, "keep_last": 96, "keep_daily": 14},
            "alert_webhook": None,
        },
        "nodes": [node("node-a", "10.77.0.1", 0, "node-a.example.net:51820"),
                  node("node-b", "10.77.0.2", 32, "node-b.example.net:51820")],
        "replication": [{"name": "vmstore", "source": "node-a", "dataset": "tank/vms",
                         "targets": ["node-b"], "interval_min": 15, "keep_last": 3}],
        "swarm": {"base_images": ["ubuntu-24.04-base.qcow2"]},
        "vms": [{"name": "web01", "disk_gib": 40, "vcpus": 2, "ram_mib": 4096,
                 "replicas": ["node-a", "node-b"], "running_on": "node-a"}],
    }
def add_wan2(data, **updates):
    peer_key = node("node-c", "10.77.0.3", 64, None)["wg_pubkey"]
    config = {
        "interface": "eth1",
        "gateway": None,
        "mtu": 1500,
        "endpoint": "node-a-wan2.example.net:51821",
        "wg1_pubkey": peer_key,
        "overlay_ipv4": "10.77.1.1",
        "uplink_mbit": 1000,
    }
    config.update(updates)
    data["nodes"][0]["wan2"] = config




class InventoryValidationTests(unittest.TestCase):
    def test_every_validation_rejection(self):
        cases = [
            (lambda d: d.update(version=2), "version must equal 1"),
            (lambda d: d["cluster"]["overlay"].update(ipv4_cidr="bad"), "valid IPv4 CIDR"),
            (lambda d: d["nodes"][0].update(name=["node-a"]), "name is invalid"),
            (lambda d: d["cluster"]["overlay"].update(plane2_ipv4_cidr="10.77.0.0/25"), "overlap"),
            (lambda d: d["nodes"][1].update(name="node-a"), "duplicate node name"),
            (lambda d: d["nodes"][1].update(overlay_ipv4="10.77.0.1"), "duplicate overlay_ipv4"),
            (lambda d: d["nodes"][1].update(overlay_ipv4="10.78.0.2"), "outside usable addresses"),
            (lambda d: d["nodes"][1].update(overlay_ipv4="10.77.0.0"), "outside usable addresses"),
            (lambda d: d["nodes"][1].update(overlay_ipv4="10.77.0.255"), "outside usable addresses"),
            (lambda d: (d["cluster"]["overlay"].update(ipv6_cidr="bad")), "valid IPv6 CIDR"),
            (lambda d: (d["cluster"]["overlay"].update(ipv6_cidr="fd77::/64"),
                        d["nodes"][0].update(overlay_ipv6="fd78::1"),
                        d["nodes"][1].update(overlay_ipv6="fd77::2")), "outside usable addresses"),
            (lambda d: (d["cluster"]["overlay"].update(ipv6_cidr="fd77::/64"),
                        d["nodes"][0].update(overlay_ipv6="fd77::1"),
                        d["nodes"][1].update(overlay_ipv6="fd77::1")), "duplicate overlay_ipv6"),
            (lambda d: d["nodes"][0].update(endpoint="node-a.example.net:65536"), "must be host:port"),
            (lambda d: add_wan2(d, mtu=1279), "wan2.mtu outside"),
            (lambda d: add_wan2(d, uplink_mbit=0), "wan2.uplink_mbit must be >= 1"),
            (lambda d: add_wan2(d, wg1_pubkey="invalid"), "wg1_pubkey"),
            (lambda d: d["cluster"]["overlay"].update(ipv6_cidr="fd77::/64"), "lacks overlay_ipv6"),
            (lambda d: d["nodes"][0].update(wg_pubkey="not-base64"), "base64 encoding"),
            (lambda d: d["nodes"][0].update(endpoint="not-an-endpoint"), "must be host:port"),
            (lambda d: [n.update(endpoint=None) for n in d["nodes"]], "no direct WireGuard path"),
            (lambda d: (d["nodes"][0].update(endpoint=None, wan2={"overlay_ipv4": "10.77.1.1", "endpoint": None,
                                                                    "wg1_pubkey": node("x", "10.77.0.3", 64, None)["wg_pubkey"],
                                                                    "mtu": 1500, "uplink_mbit": 1000,
                                                                    "interface": "eth1", "gateway": None}),
                        d["nodes"][1].update(endpoint=None)), "WAN2 and node-b: no direct WireGuard path"),
            (lambda d: d["nodes"][0].update(underlay_family="ethernet"), "underlay_family"),
            (lambda d: d["nodes"][0].update(wan_mtu=1279), "outside 1280-9000"),
            (lambda d: d["nodes"][0].update(uplink_mbit=0), "uplink_mbit must be >= 1"),
            (lambda d: d["nodes"][0].update(downlink_mbit=0), "downlink_mbit must be >= 1"),
            (lambda d: d["nodes"][0].update(uplink_mbit=1), "uplink too small"),
            (lambda d: d["nodes"][0].update(ssh_pubkey="ssh-rsa broken"), "ssh-ed25519 public key"),
            (lambda d: d["replication"][0].update(source="unknown"), "unknown source"),
            (lambda d: d["replication"].append(copy.deepcopy(d["replication"][0])), "duplicate replication job name"),
            (lambda d: d["replication"][0].update(targets=["missing"]), "unknown target"),
            (lambda d: d["replication"][0].update(targets=["node-a"]), "target equals source"),
            (lambda d: d["replication"][0].update(dataset="tank/vms;cmd"), "invalid dataset"),
            (lambda d: d["replication"][0].update(name="bad name"), "unsafe environment value"),
            (lambda d: d["replication"][0].update(targets=["node-b", "node-b"]), "duplicate target"),
            (lambda d: d["replication"][0].update(interval_min=0), "interval_min outside"),
            (lambda d: d["replication"][0].update(interval_min=1441), "interval_min outside"),
            (lambda d: d["replication"][0].update(keep_last=1), "keep_last outside"),
            (lambda d: d["nodes"][1].update(zfs={}), None),
            (lambda d: d["swarm"].update(base_images=["bad/name"]), "invalid swarm.base_images"),
            (lambda d: d["vms"][0].update(replicas=["missing"]), "unknown replica"),
            (lambda d: d["vms"][0].update(running_on="missing"), "unknown running_on"),
            (lambda d: d["vms"][0].update(disk_gib=0), "disk_gib must be positive"),
            (lambda d: d["vms"][0].update(vcpus=0), "vcpus must be positive"),
            (lambda d: d["vms"][0].update(ram_mib=0), "ram_mib must be positive"),
            (lambda d: d["cluster"]["migration"].update(port_min=49153), "power-of-two block"),
            (lambda d: d["cluster"]["migration"].update(port_min=80, port_max=143), "within 1024-65535"),
            (lambda d: d["cluster"]["migration"].update(port_max=70000), "within 1024-65535"),
            (lambda d: d["cluster"]["qos"].update(shape_percent=49), "shape_percent outside"),
            (lambda d: d["cluster"]["qos"].update(shape_percent=101), "shape_percent outside"),
            (lambda d: d["cluster"]["qos"].update(night_start="25:00"), "night_start must be HH:MM"),
            (lambda d: d["cluster"]["qos"].update(night_end="02:00"), "must differ"),
            (lambda d: d["nodes"][0]["wan"].update(interface="eth0;touch /tmp/x"), "unsafe environment"),
            (lambda d: d["cluster"].update(alert_webhook="https://example;bad"), "alert_webhook contains unsafe"),
        ]
        for index, (mutate, expected) in enumerate(cases):
            with self.subTest(case=index, expected=expected):
                data = base_inventory()
                mutate(data)
                errors, _ = inventory.validate(data)
                self.assertTrue(errors, data)
                if expected:
                    self.assertTrue(any(expected in error for error in errors), errors)

    def test_default_mtu_values(self):
        d = base_inventory()
        self.assertEqual(inventory.mtu_values(d), (60, 1440, 1440, 1390))
        d["cluster"]["overlay"]["ipv6_cidr"] = "fd77::/64"
        for n in d["nodes"]:
            n["overlay_ipv6"] = "fd77::1" if n["name"] == "node-a" else "fd77::2"
            n["underlay_family"] = "ipv6"
        self.assertEqual(inventory.mtu_values(d), (80, 1420, 1420, 1370))
        for n in d["nodes"]:
            n["wan_mtu"] = 1492
        self.assertEqual(inventory.mtu_values(d), (80, 1412, 1412, 1362))
        for n in d["nodes"]:
            n["underlay_family"] = "ipv4"
        self.assertEqual(inventory.mtu_values(d), (60, 1432, 1432, 1382))

    def test_qos_worked_values(self):
        d = base_inventory()
        self.assertEqual(inventory.rates(1000, 94, 1440, 60, 10), {
            "ROOT": 902400, "CTRL": 9600, "MIG": 192000, "REPL": 96000,
            "DEF": 595776, "BULK": 9024, "BULK_CEIL_NIGHT": 483840, "BULK_CEIL_DAY": 90240})
        self.assertEqual(inventory.rates(40, 94, 1440, 60, 10), {
            "ROOT": 36096, "CTRL": 1000, "MIG": 7680, "REPL": 3840,
            "DEF": 22576, "BULK": 1000, "BULK_CEIL_NIGHT": 18860, "BULK_CEIL_DAY": 3609})
        self.assertEqual(inventory.rates(1000, 94, 1420, 80, 10)["ROOT"], 889866)

    def test_path_worked_values(self):
        d = base_inventory()
        p = inventory.path_values(d, "node-a", "node-b")
        self.assertEqual((p["OVERLAY_TARGET_MBIT"], p["UNDERLAY_TARGET_MBIT"], p["MIG_BW_MBPS"],
                          p["DOWNTIME_MS"], p["MIG_CHANNELS"]), (826, 894, 107, 314, 4))
        for n in d["nodes"]:
            n["uplink_mbit"] = n["downlink_mbit"] = 200
        p = inventory.path_values(d, "node-a", "node-b")

    def test_wireguard_rendering_and_environment(self):
        d = base_inventory()
        d["nodes"][1]["endpoint"] = None
        d["nodes"][0]["wg_private_key"] = "SECRET_PRIVATE_KEY"
        rendered = inventory.render_wg(d, "node-a")
        self.assertNotIn("Endpoint =", rendered)
        self.assertNotIn(d["nodes"][0]["wg_private_key"], rendered)
        self.assertNotIn("private key", rendered.lower())
        self.assertNotIn("FwMark = 0x5101", rendered)
        dual = copy.deepcopy(d)
        peer_key = node("node-c", "10.77.0.3", 64, "node-c.example.net:51820")["wg_pubkey"]
        dual["nodes"][1]["wan2"] = {"interface": "eth1", "gateway": None, "mtu": 1500,
                                    "endpoint": None, "wg1_pubkey": peer_key,
                                    "overlay_ipv4": "10.77.1.2", "uplink_mbit": 40}
        dual["nodes"][0]["wan2"] = {"interface": "eth1", "gateway": None, "mtu": 1500,
                                    "endpoint": "node-a-wan2.example.net:51821", "wg1_pubkey": peer_key,
                                    "overlay_ipv4": "10.77.1.1", "uplink_mbit": 40}
        wg0 = inventory.render_wg(dual, "node-a")
        self.assertIn("FwMark = 0x5101", wg0)
        self.assertIn("# node-b (plane2)", wg0)
        self.assertIn("AllowedIPs = 10.77.1.2/32", wg0)
        wg1 = inventory.render_wg(dual, "node-a", plane2=True)
        self.assertIn("FwMark = 0x5201", wg1)
        self.assertIn("Table = off", wg1)
        self.assertNotIn("PRIVATE", wg0)
        values = inventory.env_values(dual, "node-a")
        self.assertEqual(values["P2P_WAN_GW"], "")
        self.assertEqual(values["P2P_WAN2_GW"], "")
        self.assertEqual(values["P2P_MIG_PORT_MASK"], "0xffc0")
        for value in values.values():
            self.assertRegex(str(value), r"^[A-Za-z0-9_.:/@%+,=?&~-]*$")


if __name__ == "__main__":
    unittest.main()
