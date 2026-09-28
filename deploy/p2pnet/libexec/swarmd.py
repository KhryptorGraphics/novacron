#!/usr/bin/python3
"""Private, static-peer libtorrent daemon for p2pnet."""

import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

PREFIX = Path("/opt/p2pnet")
ETC = Path("/etc/p2pnet")
STATE = Path("/var/lib/p2pnet/swarm")
PUBLIC = Path("/var/lib/p2pnet/public/torrents")
QUEUE = STATE / "queue"
RESUME = STATE / "resume"
STATUS = Path("/run/p2pnet-swarm/status.json")
stop = False
reload_inventory = False


def _signal_stop(_signum, _frame):
    global stop
    stop = True


def _signal_reload(_signum, _frame):
    global reload_inventory
    reload_inventory = True


def env_values():
    node = (ETC / "node-name").read_text(encoding="utf-8").strip()
    raw = subprocess.check_output(["/usr/bin/python3", str(PREFIX / "libexec/inventory.py"), "env", "--node", node], text=True)
    env = {}
    for line in raw.splitlines():
        key, value = line.split("=", 1)
        env[key] = json.loads(value)
    return node, env


def peers_for(node):
    raw = subprocess.check_output(["/usr/bin/python3", str(PREFIX / "libexec/inventory.py"), "peers", "--node", node], text=True)
    peers = []
    for line in raw.splitlines():
        fields = line.split("\t")
        if len(fields) != 12:
            raise RuntimeError(f"unexpected inventory peer row: {line!r}")
        address = fields[1]
        if address and address != "-":
            peers.append(address)
    return peers


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def hash_key(info):
    hashes = info.info_hashes()
    return str(hashes.v2 if hashes.has_v2() else hashes.v1)


def main():
    global reload_inventory
    try:
        import libtorrent as lt
    except ImportError as exc:
        raise SystemExit(f"python3-libtorrent is required: {exc}") from exc

    node, env = env_values()
    peers = peers_for(node)
    listen = [f"{env['P2P_OVL4']}:6881"]
    if env.get("P2P_OVL6"):
        listen.append(f"[{env['P2P_OVL6']}]:6881")
    if env.get("P2P_PLANE2_IP"):
        listen.append(f"{env['P2P_PLANE2_IP']}:6881")

    settings = {
        "listen_interfaces": ",".join(listen),
        "enable_dht": False,
        "enable_lsd": False,
        "enable_upnp": False,
        "enable_natpmp": False,
        "enable_outgoing_utp": False,
        "enable_incoming_utp": False,
        "allow_multiple_connections_per_ip": True,
        "connections_limit": 200,
        "active_downloads": 4,
        "active_seeds": -1,
        "download_rate_limit": 0,
        "upload_rate_limit": 0,
        "alert_mask": int(lt.alert.category_t.all_categories),
    }
    session = lt.session(settings)
    images = Path(env["P2P_IMAGES_DIR"])
    PUBLIC.mkdir(parents=True, exist_ok=True)
    QUEUE.mkdir(parents=True, exist_ok=True)
    RESUME.mkdir(parents=True, exist_ok=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    active = {}
    last_connect = {}
    last_status = 0
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    signal.signal(signal.SIGHUP, _signal_reload)

    while not stop:
        if reload_inventory:
            node, env = env_values()
            peers = peers_for(node)
            reload_inventory = False

        for torrent_path in sorted(PUBLIC.glob("*.torrent")):
            name = torrent_path.stem
            data = images / name
            if not data.is_file():
                continue
            try:
                info = lt.torrent_info(str(torrent_path))
                if data.stat().st_size != info.total_size():
                    continue
                info_hash = hash_key(info)
                if info_hash in active:
                    continue
                handle = session.add_torrent({"ti": info, "save_path": str(images), "flags": lt.torrent_flags.seed_mode})
                active[info_hash] = {"handle": handle, "name": name, "path": torrent_path, "queued": False}
            except (RuntimeError, OSError) as exc:
                print(f"p2pnet swarm: cannot seed {torrent_path}: {exc}", flush=True)

        for torrent_path in sorted(QUEUE.glob("*.torrent")):
            try:
                info = lt.torrent_info(str(torrent_path))
                info_hash = hash_key(info)
                if info_hash not in active:
                    resume_path = RESUME / f"{info_hash}.fastresume"
                    if resume_path.is_file():
                        try:
                            resume = lt.read_resume_data(resume_path.read_bytes())
                            resume.save_path = str(images)
                            handle = session.add_torrent(resume)
                        except (RuntimeError, TypeError):
                            handle = session.add_torrent({"ti": info, "save_path": str(images)})
                    else:
                        handle = session.add_torrent({"ti": info, "save_path": str(images)})
                    active[info_hash] = {"handle": handle, "name": info.name(), "path": torrent_path, "queued": True}
                entry = active[info_hash]
                if entry["queued"] and time.monotonic() - last_connect.get(info_hash, 0) >= 30:
                    for peer in peers:
                        entry["handle"].connect_peer((peer, 6881))
                    last_connect[info_hash] = time.monotonic()
            except (RuntimeError, OSError) as exc:
                print(f"p2pnet swarm: cannot queue {torrent_path}: {exc}", flush=True)

        alerts = session.pop_alerts()
        for alert in alerts:
            if isinstance(alert, lt.torrent_finished_alert):
                ih = hash_key(alert.handle)
                entry = active.get(ih)
                if entry and entry["queued"]:
                    try:
                        alert.handle.save_resume_data()
                    except RuntimeError:
                        pass
                    try:
                        entry["path"].unlink()
                    except FileNotFoundError:
                        pass
            elif isinstance(alert, lt.save_resume_data_alert):
                ih = hash_key(alert.handle)
                target = RESUME / f"{ih}.fastresume"
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(lt.write_resume_data_buf(alert.params))
                os.replace(temporary, target)

        now = time.time()
        if now - last_status >= 5:
            rows = []
            for ih, entry in list(active.items()):
                handle = entry["handle"]
                try:
                    s = handle.status()
                    rows.append({
                        "name": entry["name"], "infohash": ih,
                        "state": str(s.state), "progress": s.progress,
                        "download_rate": s.download_rate, "upload_rate": s.upload_rate,
                        "num_peers": s.num_peers, "total_done": s.total_done,
                        "total_wanted": s.total_wanted, "error": str(s.errc.message()) if s.errc else "",
                    })
                except RuntimeError as exc:
                    rows.append({"name": entry["name"], "infohash": ih, "state": "error", "progress": 0, "download_rate": 0, "upload_rate": 0, "num_peers": 0, "total_done": 0, "total_wanted": 0, "error": str(exc)})
            atomic_json(STATUS, {"time": now, "torrents": rows})
            last_status = now
        time.sleep(5)

    for entry in active.values():
        try:
            entry["handle"].save_resume_data()
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()
