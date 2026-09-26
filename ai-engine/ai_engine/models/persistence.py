"""Load joblib artifacts written by the AI engine services."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, TypeVar

import joblib

logger = logging.getLogger(__name__)

T = TypeVar("T")

PAYLOAD_TYPE_KEY = "payload_type"
_MAGIC = b"NOVACRON "
_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class PayloadTypeMismatch(ValueError):
    """A model file is missing its type tag or belongs to another service."""


def dump_typed_model(filepath: str, payload_type: str, model_data: Mapping[str, Any]) -> None:
    """Write ``model_data`` with a declared service type.

    The type is on the first line so a loader can reject another service's
    file before unpickling it. The same value is stored inside the payload.
    """
    _require_payload_type(payload_type)
    stamped = dict(model_data)
    stamped[PAYLOAD_TYPE_KEY] = payload_type
    buffer = io.BytesIO()
    joblib.dump(stamped, buffer)
    Path(filepath).write_bytes(_MAGIC + payload_type.encode("ascii") + b"\n" + buffer.getvalue())


def read_payload_type(filepath: str | Path) -> str | None:
    """Return the type tag, or None when the file has no tag.

    Only the first line is read. The pickled body is left untouched.
    """
    path = Path(filepath)
    with path.open("rb") as handle:
        magic = handle.read(len(_MAGIC))
        if magic != _MAGIC:
            return None
        line = handle.readline()
    try:
        payload_type = line.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if not _TYPE_RE.fullmatch(payload_type):
        return None
    return payload_type


def load_typed_model(filepath: str, expected_type: str) -> dict:
    """Unpickle a model file after its type tag matches ``expected_type``."""
    _require_payload_type(expected_type)
    found = read_payload_type(filepath)
    if found != expected_type:
        raise PayloadTypeMismatch(
            f"{Path(filepath).name}: payload_type {found!r} != {expected_type!r}"
        )
    raw = Path(filepath).read_bytes()
    newline = raw.find(b"\n")
    payload = joblib.load(io.BytesIO(raw[newline + 1 :]))
    inner = payload.get(PAYLOAD_TYPE_KEY) if isinstance(payload, dict) else None
    if inner != expected_type:
        raise PayloadTypeMismatch(
            f"{Path(filepath).name}: payload_type {inner!r} != {expected_type!r}"
        )
    return payload


def load_stored_models(
    storage_path: str,
    factory: Callable[[str], T],
    loader_name: str,
) -> Dict[str, T]:
    """Load ``*.joblib`` files whose type tag is ``loader_name``.

    A file for another service, or a file with no type tag, is skipped before
    it is unpickled. Insertion order is modification time, oldest first.
    """
    _require_payload_type(loader_name)
    loaded: Dict[str, T] = {}
    root = Path(storage_path) if storage_path else None
    if root is None or not root.is_dir():
        logger.info("%s: model directory %s is not present", loader_name, storage_path)
        return loaded

    paths = sorted(root.glob("*.joblib"), key=lambda path: path.stat().st_mtime)
    for path in paths:
        found = read_payload_type(path)
        if found != loader_name:
            logger.warning(
                "%s: skipped %s (PayloadTypeMismatch: payload_type %r != %r)",
                loader_name,
                path.name,
                found,
                loader_name,
            )
            continue
        model_id = path.stem
        model = factory(model_id)
        try:
            model.load_model(str(path))
        except Exception as exc:
            logger.warning(
                "%s: skipped %s (%s: %s)",
                loader_name,
                path.name,
                type(exc).__name__,
                exc,
            )
            continue
        if not getattr(model, "is_trained", False):
            logger.warning("%s: skipped %s (model did not become ready)", loader_name, path.name)
            continue
        metadata = getattr(model, "metadata", None)
        stored_id = getattr(metadata, "model_id", None) or model_id
        loaded[stored_id] = model

    if loaded:
        logger.info("%s: loaded %d model(s) from %s", loader_name, len(loaded), root)
    return loaded


def _require_payload_type(payload_type: str) -> None:
    if not isinstance(payload_type, str) or not _TYPE_RE.fullmatch(payload_type):
        raise ValueError(f"invalid payload_type {payload_type!r}")
