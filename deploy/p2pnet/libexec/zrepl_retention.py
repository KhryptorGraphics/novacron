#!/usr/bin/python3
"""Select old p2pnet ZFS snapshots while protecting recent and referenced names."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone

SNAPSHOT_RE = re.compile(r"@p2pnet-(\d{8}T\d{6}Z)$")


def snapshot_time(name: str) -> datetime | None:
    match = SNAPSHOT_RE.search(name)
    if match is None:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def names_to_destroy(
    names: list[str], keep_last: int, keep_daily: int, protect: set[str]
) -> list[str]:
    """Return recognized p2pnet snapshot names that are safe to destroy."""
    dated = [(snapshot_time(name), name) for name in names]
    dated = [(when, name) for when, name in dated if when is not None]
    dated.sort(key=lambda item: (item[0], item[1]), reverse=True)

    keep = set(protect)
    if dated:
        keep.add(dated[0][1])
    keep.update(name for _, name in dated[:keep_last])
    days: set[str] = set()
    for when, name in dated:
        day = when.date().isoformat()
        if day not in days:
            if len(days) < keep_daily:
                keep.add(name)
            days.add(day)

    return [name for _, name in dated if name not in keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="ZFS snapshot names")
    parser.add_argument("--keep-last", type=int, required=True)
    parser.add_argument("--keep-daily", type=int, required=True)
    parser.add_argument("--protect", action="append", default=[])
    args = parser.parse_args()
    if not 0 <= args.keep_last <= 100000 or not 0 <= args.keep_daily <= 100000:
        parser.error("keep counts must be between 0 and 100000")
    for name in names_to_destroy(args.names, args.keep_last, args.keep_daily, set(args.protect)):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
