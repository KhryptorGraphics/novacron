#!/usr/bin/python3
"""Validate and render the stable inventory contract consumed by p2pnet."""
import argparse
import base64
import ipaddress
import json
import math
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit("inventory.py requires python3-yaml (apt: python3-yaml)") from exc

SAFE_ENV = re.compile(r"^[A-Za-z0-9_.:/@%+,=?&~-]*$")
NODE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
DATASET = re.compile(r"^[A-Za-z0-9_.:-]+(?:/[A-Za-z0-9_.:-]+)*$")
IMAGE = re.compile(r"^[A-Za-z0-9._-]+$")
HOST = re.compile(r"^(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)$" )


def load(path):
    try:
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("inventory must be a YAML mapping")
    return data


def nodes_of(data):
    return data.get("nodes") if isinstance(data.get("nodes"), list) else []


def node_map(data):
    return {n.get("name"): n for n in nodes_of(data) if isinstance(n, dict) and isinstance(n.get("name"), str)}


def addr(value, version=None):
    result = ipaddress.ip_address(str(value))
    if version and result.version != version:
        raise ValueError("wrong address family")
    return result


def endpoint_ok(value):
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    match = re.fullmatch(r"(.+):(\d+)", value)
    if not match:
        return False
    host, port = match.groups()
    if not 1 <= int(port) <= 65535:
        return False
    if host.startswith("[") and host.endswith("]"):
        try:
            return ipaddress.ip_address(host[1:-1]).version == 6
        except ValueError:
            return False
    try:
        ipaddress.IPv4Address(host)
        return True
    except ValueError:
        labels = host.rstrip(".").split(".")
        valid_hostname = (len(host) <= 253 and all(
            re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
            for label in labels))
        return valid_hostname and not host.startswith("[")

def valid_wg_key(value):
    try:
        return isinstance(value, str) and len(base64.b64decode(value, validate=True)) == 32
    except (ValueError, TypeError):
        return False


def valid_ssh_key(value):
    return isinstance(value, str) and bool(re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/]+={0,2}(?: .*)?", value))


def overhead(data):
    return 60 if all(n.get("underlay_family", "ipv4") == "ipv4" for n in nodes_of(data)) else 80


def mtu_values(data):
    ns = nodes_of(data)
    ov = data["cluster"]["overlay"]
    extra = overhead(data)
    wg = min(int(n["wan_mtu"]) for n in ns) - extra
    dual_mtus = [int(n["wan2"].get("mtu", n["wan_mtu"])) for n in ns if isinstance(n.get("wan2"), dict)]
    wg1 = min([int(n["wan_mtu"]) for n in ns] + dual_mtus) - extra
    return extra, wg, wg1, wg - 50


def rates(uplink, shape, wg_mtu, extra, day_percent):
    outer = int(uplink) * 1000 * int(shape) // 100
    root = outer * wg_mtu // (wg_mtu + extra)
    ctrl = max(1000, root * 10 // 940)
    mig = root * 200 // 940
    repl = root * 100 // 940
    bulk = max(1000, root // 100)
    default = root - ctrl - mig - repl - bulk
    night = max(bulk, 8 * (root - ctrl - mig - repl) // 10)
    day = max(bulk, root * int(day_percent) // 100)
    return {"ROOT": root, "CTRL": ctrl, "MIG": mig, "REPL": repl, "DEF": default,
            "BULK": bulk, "BULK_CEIL_NIGHT": night, "BULK_CEIL_DAY": day}


def path_values(data, source, target):
    ns = node_map(data)
    a, b = ns[source], ns[target]
    c = data["cluster"]
    extra, wg, _, _ = mtu_values(data)
    shape = int(c.get("qos", {}).get("shape_percent", 94))
    root = rates(a["uplink_mbit"], shape, wg, extra, c.get("qos", {}).get("bulk_day_ceil_percent", 10))["ROOT"]
    dst = int(b["downlink_mbit"]) * 1000 * shape // 100 * wg // (wg + extra)
    path = min(root, dst)
    ceiling = path * (wg - 52) // wg // 1000
    target_rate = path * (wg - 52) * 95 // (wg * 100 * 1000)
    underlay = min(int(a["uplink_mbit"]), int(b["downlink_mbit"])) * 1448 * 95 // (1538 * 100)
    bw = path * 95 // 100 // 8000
    downtime = min(2000, max(300, math.ceil(33554.432 / bw)))
    return {"PATH_KBIT": path, "OVERLAY_CEIL_MBIT": ceiling, "OVERLAY_TARGET_MBIT": target_rate,
            "UNDERLAY_TARGET_MBIT": underlay, "MIG_BW_MBPS": bw, "DOWNTIME_MS": downtime,
            "MIG_CHANNELS": 4 if int(a["uplink_mbit"]) <= 1000 else 8,
            "DUALWAN": int(bool(a.get("wan2") or b.get("wan2")))}


def validate(data):
    errors, warnings = [], []
    def err(message): errors.append(message)
    if data.get("version") != 1: err("version must equal 1")
    cluster = data.get("cluster") if isinstance(data.get("cluster"), dict) else {}
    overlay = cluster.get("overlay") if isinstance(cluster.get("overlay"), dict) else {}
    try:
        net4 = ipaddress.ip_network(overlay.get("ipv4_cidr"), strict=True)
        if net4.version != 4: raise ValueError()
    except (ValueError, TypeError):
        net4 = None; err("cluster.overlay.ipv4_cidr is not a valid IPv4 CIDR")
    net2 = None
    if overlay.get("plane2_ipv4_cidr") is not None:
        try:
            net2 = ipaddress.ip_network(overlay["plane2_ipv4_cidr"], strict=True)
            if net2.version != 4: raise ValueError()
            if net4 and net2.overlaps(net4): err("ipv4_cidr and plane2_ipv4_cidr overlap")
        except (ValueError, TypeError): err("cluster.overlay.plane2_ipv4_cidr is not a valid IPv4 CIDR")
    net6 = None
    if overlay.get("ipv6_cidr") is not None:
        try:
            net6 = ipaddress.ip_network(overlay["ipv6_cidr"], strict=True)
            if net6.version != 6: raise ValueError()
        except (ValueError, TypeError): err("cluster.overlay.ipv6_cidr is not a valid IPv6 CIDR")
    ns = nodes_of(data)
    if not ns: err("nodes must be a non-empty list")
    names, ips4, ips6, ips2 = set(), set(), set(), set()
    for i, n in enumerate(ns):
        if not isinstance(n, dict): err(f"nodes[{i}] must be a mapping"); continue
        name = n.get("name")
        if not isinstance(name, str) or not NODE_NAME.fullmatch(name):
            err(f"nodes[{i}].name is invalid")
        else:
            if name in names: err(f"duplicate node name {name}")
            names.add(name)
        for key, version, network, seen in (("overlay_ipv4", 4, net4, ips4), ("overlay_ipv6", 6, net6, ips6)):
            value = n.get(key)
            if key == "overlay_ipv6" and value is None and net6 is None: continue
            if value is None:
                if key == "overlay_ipv6" and net6 is not None: err(f"{name} lacks overlay_ipv6")
                elif key == "overlay_ipv4": err(f"{name} lacks overlay_ipv4")
                continue
            try:
                ip = addr(value, version)
                if ip in seen: err(f"duplicate {key} {ip}")
                seen.add(ip)
                if network and (ip not in network or ip in (network.network_address, network.broadcast_address)):
                    err(f"{name}.{key} is outside usable addresses of its CIDR")
            except ValueError: err(f"{name}.{key} is not IPv{version}")
        if net2 and isinstance(n.get("wan2"), dict):
            w2 = n["wan2"]
            try:
                ip = addr(w2.get("overlay_ipv4"), 4)
                if ip in ips2: err(f"duplicate plane2 overlay_ipv4 {ip}")
                ips2.add(ip)
                if ip not in net2 or ip in (net2.network_address, net2.broadcast_address): err(f"{name}.wan2.overlay_ipv4 is outside usable plane2 CIDR")
            except ValueError: err(f"{name}.wan2.overlay_ipv4 is invalid")
        if not valid_wg_key(n.get("wg_pubkey")): err(f"{name}.wg_pubkey must be base64 encoding of 32 bytes")
        if n.get("endpoint") is not None and not endpoint_ok(n.get("endpoint")): err(f"{name}.endpoint must be host:port")
        if n.get("underlay_family", "ipv4") not in ("ipv4", "ipv6"): err(f"{name}.underlay_family must be ipv4 or ipv6")
        for key in ("ssh_pubkey", "ssh_bulk_pubkey", "ssh_hostkey"):
            if not valid_ssh_key(n.get(key)): err(f"{name}.{key} must be an ssh-ed25519 public key")
        for key in ("wan_mtu", "uplink_mbit", "downlink_mbit"):
            try: value = int(n[key])
            except (KeyError, TypeError, ValueError): value = 0
            if key == "wan_mtu" and not 1280 <= value <= 9000: err(f"{name}.{key} outside 1280-9000")
            if key != "wan_mtu" and value < 1: err(f"{name}.{key} must be >= 1")
        try: node_up = int(n.get("uplink_mbit", 0) or 0)
        except (TypeError, ValueError): node_up = 0
        if node_up > 0:
            try:
                _, wgm, _, _ = mtu_values(data)
                q = rates(node_up, cluster.get("qos", {}).get("shape_percent", 94), wgm, overhead(data), cluster.get("qos", {}).get("bulk_day_ceil_percent", 10))
                if q["DEF"] < 1000: err(f"{name} uplink too small: computed DEF rate below 1000 kbit")
            except (KeyError, ValueError, TypeError): pass
        wan = n.get("wan") or {}
        if isinstance(n.get("wan2"), dict):
            w2 = n["wan2"]
            try: mtu2 = int(w2.get("mtu", n.get("wan_mtu", 0)))
            except (ValueError, TypeError): mtu2 = 0
            if not 1280 <= mtu2 <= 9000: err(f"{name}.wan2.mtu outside 1280-9000")
            if w2.get("gateway") is None: warnings.append(f"{name} wan2.gateway is null; only on-link endpoints are reachable through WAN2")
            if w2.get("endpoint") is not None and not endpoint_ok(w2.get("endpoint")): err(f"{name}.wan2.endpoint must be host:port")
            if not valid_wg_key(w2.get("wg1_pubkey")): err(f"{name}.wan2.wg1_pubkey must be base64 encoding of 32 bytes")
            try:
                if int(w2.get("uplink_mbit", 0)) < 1: err(f"{name}.wan2.uplink_mbit must be >= 1")
                else:
                    q = rates(w2["uplink_mbit"], cluster.get("qos", {}).get("shape_percent", 94), min(w2.get("mtu", n["wan_mtu"]), min(int(x["wan_mtu"]) for x in ns if isinstance(x, dict))), overhead(data), cluster.get("qos", {}).get("bulk_day_ceil_percent", 10))
                    if q["DEF"] < 1000: err(f"{name} wan2 uplink too small: computed DEF rate below 1000 kbit")
            except (ValueError, TypeError, KeyError): pass
        for key, value in (("wan.interface", wan.get("interface")), ("wan.gateway", wan.get("gateway")), ("images_dir", n.get("images_dir")), ("zfs.replica_root", (n.get("zfs") or {}).get("replica_root"))):
            if value is not None and not SAFE_ENV.fullmatch(str(value)): err(f"{name}.{key} contains unsafe environment characters")
    for i, a in enumerate(ns):
        for b in ns[i+1:]:
            if a.get("endpoint") is None and b.get("endpoint") is None: err(f"{a.get('name')} and {b.get('name')}: no direct WireGuard path")
    for a in ns:
        if not isinstance(a.get("wan2"), dict): continue
        for b in ns:
            if a is b: continue
            if a["wan2"].get("endpoint") is None and b.get("endpoint") is None: err(f"{a.get('name')} WAN2 and {b.get('name')}: no direct WireGuard path")
    qos = cluster.get("qos", {}) or {}
    try:
        shape = int(qos.get("shape_percent", 94))
        if not 50 <= shape <= 100: err("shape_percent outside 50-100")
        day = int(qos.get("bulk_day_ceil_percent", 10))
        if not 1 <= day <= 100: err("bulk_day_ceil_percent outside 1-100")
    except (TypeError, ValueError): err("QoS percentages must be integers")
    def clock(value): return isinstance(value, str) and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value) is not None
    start, end = qos.get("night_start", "02:00"), qos.get("night_end", "06:00")
    if not clock(start): err("night_start must be HH:MM")
    if not clock(end): err("night_end must be HH:MM")
    if start == end: err("night_start and night_end must differ")
    migration = cluster.get("migration", {}) or {}
    try:
        lo, hi = int(migration.get("port_min", 49152)), int(migration.get("port_max", 49215))
        size = hi - lo + 1
        if not (1024 <= lo <= hi <= 65535) or size & (size - 1) or lo % size: err("migration range must be a power-of-two block aligned to its size within 1024-65535")
    except (TypeError, ValueError): err("migration port range must contain integer port_min and port_max")
    defaults = cluster.get("replication_defaults", {}) or {}
    jobs, jobnames = data.get("replication", []) or [], set()
    for j in jobs:
        if not isinstance(j, dict): err("replication entries must be mappings"); continue
        name = j.get("name")
        if isinstance(name, str) and not SAFE_ENV.fullmatch(name): err(f"replication {name}: unsafe environment value")
        if not isinstance(name, str) or not name: err("replication job has invalid name")
        elif name in jobnames: err(f"duplicate replication job name {name}")
        jobnames.add(name)
        src = j.get("source"); targets = j.get("targets", [])
        if src not in names: err(f"replication {name}: unknown source {src}")
        if not isinstance(j.get("dataset"), str) or not DATASET.fullmatch(j.get("dataset", "")): err(f"replication {name}: invalid dataset")
        if not isinstance(targets, list): targets = []; err(f"replication {name}: targets must be a list")
        if any(targets.count(target) > 1 for target in targets): err(f"replication {name}: duplicate target")
        for target in targets:
            if target not in names: err(f"replication {name}: unknown target {target}")
            if target == src: err(f"replication {name}: target equals source")
            elif target in node_map(data) and not (node_map(data)[target].get("zfs") or {}).get("replica_root"): err(f"replication {name}: target {target} has no zfs.replica_root")
        for key, default, low in (("interval_min", 15, 1), ("keep_last", 96, 2)):
            try: value = int(j.get(key, defaults.get(key, default)))
            except (TypeError, ValueError): value = 0
            if value < low or (key == "interval_min" and value > 1440):
                err(f"replication {name}: {key} outside {low}-{'1440' if key == 'interval_min' else 'infinity'}")
        if isinstance(targets, list) and all(isinstance(target, str) for target in targets):
            values = [name, src, j.get("dataset", ""), ",".join(targets),
                      j.get("interval_min", defaults.get("interval_min", 15)),
                      j.get("keep_last", defaults.get("keep_last", 96)),
                      j.get("keep_daily", defaults.get("keep_daily", 14))]
            for value in values:
                if not SAFE_ENV.fullmatch(str(value)):
                    err(f"replication {name}: unsafe environment value")
    for image in (data.get("swarm") or {}).get("base_images", []) or []:
        if not isinstance(image, str) or not IMAGE.fullmatch(image): err(f"invalid swarm.base_images name {image}")
    for vm in data.get("vms", []) or []:
        if not isinstance(vm, dict): err("VM entries must be mappings"); continue
        for node in (vm.get("replicas") or []):
            if node not in names: err(f"VM {vm.get('name')}: unknown replica {node}")
        if vm.get("running_on") is not None and vm.get("running_on") not in names: err(f"VM {vm.get('name')}: unknown running_on node")
        for key in ("disk_gib", "vcpus", "ram_mib"):
            try: value = int(vm.get(key, 0))
            except (TypeError, ValueError): value = 0
            if value <= 0: err(f"VM {vm.get('name')}: {key} must be positive")
    for n in ns:
        if not isinstance(n, dict) or not isinstance(n.get("name"), str) or n.get("name") not in node_map(data):
            continue
        try:
            for key, value in env_values(data, n["name"]).items():
                if not SAFE_ENV.fullmatch(str(value)):
                    err(f"{n['name']}.{key} contains unsafe environment characters")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
    webhook = cluster.get("alert_webhook")
    if webhook is not None and not SAFE_ENV.fullmatch(str(webhook)): err("alert_webhook contains unsafe environment characters")
    return errors, warnings


def require_node(data, name):
    if name not in node_map(data): raise ValueError(f"unknown node {name}")
    return node_map(data)[name]


def env_values(data, name):
    n = require_node(data, name); c = data["cluster"]; o = c["overlay"]
    ns = nodes_of(data); idx = next(i for i, x in enumerate(ns) if x["name"] == name)
    extra, wg, wg1, vx = mtu_values(data)
    v4net = ipaddress.ip_network(o["ipv4_cidr"]); v4 = addr(n["overlay_ipv4"])
    v6net = ipaddress.ip_network(o["ipv6_cidr"]) if o.get("ipv6_cidr") else None
    v6 = addr(n["overlay_ipv6"]) if n.get("overlay_ipv6") else None
    plane = ipaddress.ip_network(o["plane2_ipv4_cidr"]) if o.get("plane2_ipv4_cidr") else None
    plane_ip = addr(n["wan2"]["overlay_ipv4"]) if n.get("wan2") else None
    qos = c.get("qos", {})
    q = rates(n["uplink_mbit"], qos.get("shape_percent", 94), wg, extra, qos.get("bulk_day_ceil_percent", 10))
    q2 = rates(n["wan2"]["uplink_mbit"], qos.get("shape_percent", 94), min(wg1, int(n["wan2"].get("mtu", n["wan_mtu"])) - extra), extra, qos.get("bulk_day_ceil_percent", 10)) if n.get("wan2") else None
    migration = c.get("migration", {})
    lo, hi = int(migration.get("port_min", 49152)), int(migration.get("port_max", 49215))
    def quote(v):
        s = "" if v is None else str(v)
        if not SAFE_ENV.fullmatch(s): raise ValueError(f"unsafe environment value {s!r}")
        return f'"{s}"'
    values = {
        "P2P_CLUSTER": c.get("name", ""), "P2P_NODE": name, "P2P_NODE_INDEX": idx,
        "P2P_WG_PUBKEY": n.get("wg_pubkey", ""), "P2P_WG1_PUBKEY": (n.get("wan2") or {}).get("wg1_pubkey", ""),
        "P2P_OVL4": str(v4), "P2P_OVL4_PREFIX": v4net.prefixlen, "P2P_OVL4_CIDR": str(v4net),
        "P2P_OVL6": str(v6) if v6 else "", "P2P_OVL6_PREFIX": v6net.prefixlen if v6net else "",
        "P2P_PLANE2_CIDR": str(plane) if plane else "", "P2P_PLANE2_IP": str(plane_ip) if plane_ip else "",
        "P2P_WG_PORT": o.get("wg_port", 51820), "P2P_WG1_PORT": o.get("wg1_port", 51821),
        "P2P_WG_OVERHEAD": extra, "P2P_WG_MTU": wg, "P2P_WG1_MTU": wg1,
        "P2P_VXLAN_VNI": c.get("vxlan", {}).get("vni", 7700), "P2P_VXLAN_PORT": c.get("vxlan", {}).get("port", 4789),
        "P2P_VXLAN_MTU": vx, "P2P_BRIDGE": c.get("vxlan", {}).get("bridge", "br-p2p"),
        "P2P_WAN_IF": (n.get("wan") or {}).get("interface") or "", "P2P_WAN_GW": (n.get("wan") or {}).get("gateway") or "",
        "P2P_WAN2_IF": (n.get("wan2") or {}).get("interface") or "", "P2P_WAN2_GW": (n.get("wan2") or {}).get("gateway") or "",
        "P2P_UPLINK_MBIT": n["uplink_mbit"], "P2P_DOWNLINK_MBIT": n["downlink_mbit"],
        "P2P_NIGHT_START": qos.get("night_start", "02:00"), "P2P_NIGHT_END": qos.get("night_end", "06:00"),
        "P2P_MIG_PORT_MIN": lo, "P2P_MIG_PORT_MAX": hi, "P2P_MIG_PORT_MASK": hex(0xffff & ~(hi-lo)),
        "P2P_REPLICA_ROOT": (n.get("zfs") or {}).get("replica_root", ""), "P2P_IMAGES_DIR": n.get("images_dir", "/var/lib/p2pnet/images"),
        "P2P_ALERT_WEBHOOK": c.get("alert_webhook") or "",
    }
    for key, value in q.items(): values[f"P2P_QOS_{key}_KBIT"] = value
    for key in ("ROOT", "CTRL", "MIG", "REPL", "DEF", "BULK", "BULK_CEIL_NIGHT", "BULK_CEIL_DAY"):
        values[f"P2P_QOS2_{key}_KBIT"] = q2[key] if q2 else ""
    return values


def env_lines(values):
    for key, value in values.items():
        s = str(value)
        if not SAFE_ENV.fullmatch(s): raise ValueError(f"unsafe value for {key}: {s!r}")
        print(f"{key}={json.dumps(s)}")


def peers(data, name):
    require_node(data, name)
    for n in nodes_of(data):
        if n["name"] == name: continue
        w2 = n.get("wan2") or {}
        fields = [n["name"], n.get("overlay_ipv4", ""), n.get("overlay_ipv6") or "-", n.get("endpoint") or "-",
                  n.get("wg_pubkey", ""), w2.get("overlay_ipv4") or "-", w2.get("wg1_pubkey") or "-",
                  str(next(i for i,x in enumerate(nodes_of(data)) if x["name"]==n["name"])), str(n.get("uplink_mbit", "")),
                  str(n.get("downlink_mbit", "")), str(int(bool(n.get("underlay_iperf", False)))),
                  (n.get("zfs") or {}).get("replica_root") or "-"]
        print("\t".join(fields))


def render_wg(data, name, plane2=False):
    n = require_node(data, name); env = env_values(data, name); o = data["cluster"]["overlay"]
    if plane2:
        if not n.get("wan2"): return ""
        out = ["# Generated by p2pnet from /etc/p2pnet/inventory.yaml — edit the inventory, not this file.", "[Interface]",
               f"Address = {n['wan2']['overlay_ipv4']}/{ipaddress.ip_network(o['plane2_ipv4_cidr']).prefixlen}",
               f"ListenPort = {o.get('wg1_port', 51821)}", f"MTU = {env['P2P_WG1_MTU']}", "FwMark = 0x5201", "Table = off",
               "PostUp = wg set %i private-key /etc/wireguard/%i.key", "PostUp = /opt/p2pnet/libexec/plane2-routes up",
               "PreDown = /opt/p2pnet/libexec/plane2-routes down", "SaveConfig = false"]
        for peer in data["nodes"]:
            if peer["name"] == name: continue
            out.extend(["", "[Peer]", f"# {peer['name']}", f"PublicKey = {peer['wg_pubkey']}"])
            if peer.get("endpoint"): out.append(f"Endpoint = {peer['endpoint']}")
            out.extend([f"AllowedIPs = {peer['overlay_ipv4']}/32", "PersistentKeepalive = 25"])
        return "\n".join(out) + "\n"
    addrstr = f"{n['overlay_ipv4']}/{env['P2P_OVL4_PREFIX']}"
    if n.get("overlay_ipv6"): addrstr += f", {n['overlay_ipv6']}/{env['P2P_OVL6_PREFIX']}"
    out = ["# Generated by p2pnet from /etc/p2pnet/inventory.yaml — edit the inventory, not this file.", "[Interface]",
           f"Address = {addrstr}", f"ListenPort = {o.get('wg_port', 51820)}", f"MTU = {env['P2P_WG_MTU']}"]
    if n.get("wan2"): out.append("FwMark = 0x5101")
    out.extend(["PostUp = wg set %i private-key /etc/wireguard/%i.key", "SaveConfig = false"])
    for peer in data["nodes"]:
        if peer["name"] == name: continue
        allowed = f"{peer['overlay_ipv4']}/32"
        if peer.get("overlay_ipv6"): allowed += f", {peer['overlay_ipv6']}/128"
        out.extend(["", "[Peer]", f"# {peer['name']}", f"PublicKey = {peer['wg_pubkey']}"])
        if peer.get("endpoint"): out.append(f"Endpoint = {peer['endpoint']}")
        out.extend([f"AllowedIPs = {allowed}", "PersistentKeepalive = 25"])
        if peer.get("wan2"):
            w2=peer["wan2"]
            out.extend(["", "[Peer]", f"# {peer['name']} (plane2)", f"PublicKey = {w2['wg1_pubkey']}"])
            if w2.get("endpoint"): out.append(f"Endpoint = {w2['endpoint']}")
            out.extend([f"AllowedIPs = {w2['overlay_ipv4']}/32", "PersistentKeepalive = 25"])
    return "\n".join(out) + "\n"


def render_known_hosts(data):
    for n in nodes_of(data):
        hosts = [n["overlay_ipv4"], f"[{n['overlay_ipv4']}]:2222", f"[{n['overlay_ipv4']}]:2223"]
        if n.get("overlay_ipv6"):
            hosts += [n["overlay_ipv6"], f"[{n['overlay_ipv6']}]:2222", f"[{n['overlay_ipv6']}]:2223"]
        if n.get("wan2"):
            p=n["wan2"]["overlay_ipv4"]; hosts += [p, f"[{p}]:2222", f"[{p}]:2223"]
        print(",".join(hosts) + " " + n["ssh_hostkey"])


def authorized_keys(data, name, user):
    require_node(data, name)
    if user not in ("p2prepl", "p2pbulk", "p2pvirt"): raise ValueError("user must be p2prepl, p2pbulk or p2pvirt")
    for peer in nodes_of(data):
        if peer["name"] == name: continue
        froms=[peer["overlay_ipv4"]]
        if peer.get("overlay_ipv6"): froms.append(peer["overlay_ipv6"])
        if peer.get("wan2"): froms.append(peer["wan2"]["overlay_ipv4"])
        key=peer["ssh_bulk_pubkey"] if user=="p2pbulk" else peer["ssh_pubkey"]
        prefix='restrict'
        if user=="p2prepl": prefix += f',command="/opt/p2pnet/libexec/zrepl-guard {peer["name"]}"'
        prefix += ',from="' + ",".join(froms) + '"'
        print(f"{prefix} {key}")


def jobs(data, name):
    require_node(data, name); defaults=data["cluster"].get("replication_defaults", {})
    for j in data.get("replication", []) or []:
        if j.get("source") != name: continue
        targets=",".join(j.get("targets", []))
        fields=[j["name"],j["dataset"],targets,str(j.get("interval_min",defaults.get("interval_min",15))),
                str(j.get("keep_last",defaults.get("keep_last",96))),str(j.get("keep_daily",defaults.get("keep_daily",14)))]
        print("\t".join(fields))


def job_env(data, name):
    matches=[j for j in (data.get("replication", []) or []) if j.get("name")==name]
    if not matches: raise ValueError(f"unknown replication job {name}")
    j=matches[0]; d=data["cluster"].get("replication_defaults", {})
    env_lines({"P2P_JOB":j["name"],"P2P_JOB_SOURCE":j["source"],"P2P_JOB_DATASET":j["dataset"],
               "P2P_JOB_TARGETS":",".join(j.get("targets",[])),"P2P_JOB_INTERVAL_MIN":j.get("interval_min",d.get("interval_min",15)),
               "P2P_JOB_KEEP_LAST":j.get("keep_last",d.get("keep_last",96)),"P2P_JOB_KEEP_DAILY":j.get("keep_daily",d.get("keep_daily",14))})


def placement(data, vm):
    nodes=node_map(data); vms=data.get("vms", []) or []
    if isinstance(vm,str): vm=next((v for v in vms if v.get("name")==vm),None)
    if vm is None: raise ValueError("unknown VM")
    used={name:[0,0] for name in nodes}
    for other in vms:
        running=other.get("running_on")
        if running in used and other is not vm:
            used[running][0]+=int(other.get("vcpus",0)); used[running][1]+=int(other.get("ram_mib",0))
    need_cpu,need_ram=int(vm["vcpus"]),int(vm["ram_mib"])
    fits=[]
    for name,n in nodes.items():
        cap=n.get("capacity",{}) or {}
        free_cpu=int(cap.get("vcpus",0))-used[name][0]; free_ram=int(cap.get("ram_mib",0))-used[name][1]
        if free_cpu>=need_cpu and free_ram>=need_ram: fits.append((name,free_cpu-need_cpu,free_ram-need_ram))
    run=vm.get("running_on"); replicas=vm.get("replicas",[]) or []
    if run in replicas and any(x[0]==run for x in fits): return {"vm":vm["name"],"node":run,"action":"stay","wan_copy_gib":0,"reason":"already running on a replica with capacity"}
    for preferred in replicas:
        if any(x[0]==preferred for x in fits): return {"vm":vm["name"],"node":preferred,"action":"move","wan_copy_gib":0,"reason":"first replica with capacity"}
    if fits:
        name,_,_=sorted(fits,key=lambda x:(-x[2],-int(nodes[x[0]].get("uplink_mbit",0)),x[0]))[0]
        return {"vm":vm["name"],"node":name,"action":"copy","wan_copy_gib":int(vm["disk_gib"]),"reason":"best remaining capacity requires full copy"}
    raise ValueError(f"no node has capacity for {vm['name']}")


def place_cli(data,args):
    vms=data.get("vms",[]) or []
    selected=vms if args.all else [next((v for v in vms if v.get("name")==args.vm),None)]
    rows=[placement(data,v) for v in selected]
    if args.json:
        print(json.dumps(rows if args.all else rows[0],sort_keys=True)); return
    print("VM\tNODE\tACTION\tWAN_COPY_GIB\tREASON")
    for r in rows: print(f"{r['vm']}\t{r['node']}\t{r['action']}\t{r['wan_copy_gib']}\t{r['reason']}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory",default=os.environ.get("P2PNET_INVENTORY","/etc/p2pnet/inventory.yaml"))
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("validate")
    for command in ("env","peers","render-wg0","render-wg1","jobs","qos-table"):
        p=sub.add_parser(command); p.add_argument("--node",required=True)
    p=sub.add_parser("path"); p.add_argument("--from",dest="source",required=True); p.add_argument("--to",dest="target",required=True)
    sub.add_parser("render-known-hosts")
    p=sub.add_parser("render-authorized-keys"); p.add_argument("--node",required=True); p.add_argument("--user",required=True)
    p=sub.add_parser("job"); p.add_argument("--name",required=True)
    p=sub.add_parser("place"); p.add_argument("vm",nargs="?"); p.add_argument("--all",action="store_true"); p.add_argument("--json",action="store_true")
    p=sub.add_parser("render-all"); p.add_argument("--out",required=True)
    args=parser.parse_args()
    try:
        data=load(args.inventory)
        if args.command=="validate":
            errors,warnings=validate(data)
            for warning in warnings: print("WARN " + warning)
            for error in errors: print("ERROR " + error)
            if not errors: print("inventory valid")
            return 1 if errors else 0
        errors,_=validate(data)
        if errors: raise ValueError("inventory invalid: " + "; ".join(errors))
        if args.command=="env": env_lines(env_values(data,args.node))
        elif args.command=="path":
            for k,v in path_values(data,args.source,args.target).items():
                suffix = k[5:] if k.startswith("PATH_") else k
                print(f"P2P_PATH_{suffix}={v}")
        elif args.command=="peers": peers(data,args.node)
        elif args.command=="render-wg0": print(render_wg(data,args.node),end="")
        elif args.command=="render-wg1": print(render_wg(data,args.node,True),end="")
        elif args.command=="render-known-hosts": render_known_hosts(data)
        elif args.command=="render-authorized-keys": authorized_keys(data,args.node,args.user)
        elif args.command=="jobs": jobs(data,args.node)
        elif args.command=="job": job_env(data,args.name)
        elif args.command=="qos-table":
            e,w,g,_=mtu_values(data); n=require_node(data,args.node); q=rates(n["uplink_mbit"],data["cluster"].get("qos",{}).get("shape_percent",94),w,e,data["cluster"].get("qos",{}).get("bulk_day_ceil_percent",10))
            print("CLASS\tRATE_KBIT\tNIGHT_CEIL_KBIT\tDAY_CEIL_KBIT")
            for key in ("CTRL","MIG","REPL","DEF","BULK"): print(f"{key}\t{q[key]}\t{q['BULK_CEIL_NIGHT'] if key=='BULK' else q['ROOT']}\t{q['BULK_CEIL_DAY'] if key=='BULK' else q['ROOT']}")
            print(f"ROOT\t{q['ROOT']}\t{q['ROOT']}\t{q['ROOT']}")
        elif args.command=="place":
            if not args.all and not args.vm: raise ValueError("place requires VM or --all")
            place_cli(data,args)
        elif args.command=="render-all":
            out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
            for n in nodes_of(data):
                d=out/n["name"]; d.mkdir(parents=True,exist_ok=True)
                (d/"wg0.conf").write_text(render_wg(data,n["name"]),encoding="utf-8")
                (d/"wg1.conf").write_text(render_wg(data,n["name"],True),encoding="utf-8")
                (d/"node.env").write_text("".join(f"{k}={json.dumps(str(v))}\n" for k,v in env_values(data,n["name"]).items()),encoding="utf-8")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"ERROR {exc}",file=sys.stderr); return 1

if __name__=="__main__": sys.exit(main())
