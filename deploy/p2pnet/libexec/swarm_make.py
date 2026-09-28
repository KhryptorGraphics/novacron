#!/usr/bin/python3
"""Build private v2-only torrents for p2pnet image distribution."""

import argparse
import os
from pathlib import Path

MIB = 1024 * 1024
GIB = 1024 * MIB


def piece_size_for(size_bytes):
    if size_bytes < 0:
        raise ValueError("size must be non-negative")
    if size_bytes <= 64 * GIB:
        return 16 * MIB
    if size_bytes <= 256 * GIB:
        return 32 * MIB
    return 64 * MIB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("name")
    args = parser.parse_args()
    if not args.file.is_file():
        parser.error(f"not a regular file: {args.file}")
    if not args.name or "/" in args.name or args.name in (".", ".."):
        parser.error("name must be a single filename")
    try:
        import libtorrent as lt
    except ImportError as exc:
        parser.error(f"python3-libtorrent is required: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fs = lt.file_storage()
    lt.add_files(fs, str(args.file))
    # Preserve the requested published name rather than the source path basename.
    fs.rename_file(0, args.name)
    torrent = lt.create_torrent(fs, piece_size_for(args.file.stat().st_size),
                                flags=lt.create_torrent.v2_only)
    torrent.set_priv(True)
    torrent.set_creator("p2pnet")
    torrent.set_comment(args.name)
    lt.set_piece_hashes(torrent, str(args.file.parent))
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_bytes(lt.bencode(torrent.generate()))
    os.replace(temporary, args.out)


if __name__ == "__main__":
    main()
