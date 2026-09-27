"""Timezone-aware UTC clock helpers.

The deprecated naive-UTC ``datetime`` constructor returns a naive value
that silently represents UTC, inviting naive/aware comparison bugs. These
helpers replace every such call site in the AI engine with
``datetime.now(timezone.utc)`` while preserving each site's pre-existing
wire format:

- ``utc_now()`` -- full timezone-aware instant, for values only ever
  compared against other values produced the same way (never serialized
  to a string another process parses back).
- ``utc_now_naive()`` -- the same instant with ``tzinfo`` stripped, for
  fields whose JSON wire format predates timezone-awareness (a bare ISO
  string with no UTC offset) and must not change shape.
- ``utc_now_iso_z()`` -- an RFC 3339 string with a single trailing ``Z``,
  for fields that already appended ``"Z"`` manually (appending ``"Z"`` to
  an aware ``isoformat()`` result would otherwise double up as
  ``"+00:00Z"``).
"""

from datetime import datetime, timezone

__all__ = ["utc_now", "utc_now_naive", "utc_now_iso_z"]


def utc_now() -> datetime:
    """Timezone-aware current UTC instant."""
    return datetime.now(timezone.utc)


def utc_now_naive() -> datetime:
    """UTC instant as a naive ``datetime``, matching pre-existing wire
    shapes that never carried a UTC offset."""
    return utc_now().replace(tzinfo=None)


def utc_now_iso_z() -> str:
    """RFC 3339 string for the current UTC instant with a trailing ``Z``."""
    return utc_now().isoformat().replace("+00:00", "Z")
