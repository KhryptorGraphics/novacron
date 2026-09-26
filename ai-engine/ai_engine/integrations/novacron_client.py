"""Async client for the NovaCron API, shared by all AI-engine services.

The client owns authentication (login with token caching and single re-login
on 401) and bounded retries. Metric row mappers turn the canonical NovaCron
monitoring payloads into the feature shapes the models were trained on:

- Failure prediction / anomaly detection: rows with ``node_id``, ``timestamp``
  and measured ``cpu_usage`` / ``memory_usage`` / ``disk_usage`` /
  ``network_usage`` keys. Keys are omitted when the backend has no sample
  rather than filled with fabricated values.
- Resource optimization: one host-level row with 0-1 utilization fractions,
  which is the only resource the backend instruments across all three of
  cpu/memory/storage.

Empty results are returned as ``None`` so monitoring loops back off instead of
predicting on nothing.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

from ..config import NovaCronSettings


logger = logging.getLogger(__name__)


class NovaCronAPIError(RuntimeError):
    """Raised when the NovaCron API cannot be reached or rejects requests."""


_FRACTION_TRIM = re.compile(r"(\.\d{6})\d+")


def _parse_rfc3339(value: Any) -> Optional[datetime]:
    """Parse Go-style RFC3339 timestamps (up to nanosecond fractions)."""
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    # Go emits up to 9 fractional digits; datetime only keeps microseconds.
    text = _FRACTION_TRIM.sub(r"\1", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class NovaCronClient:
    """Thin async wrapper over the NovaCron API used by the monitoring loops."""

    def __init__(self, settings: NovaCronSettings,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.api_url,
            timeout=settings.api_timeout,
            transport=transport,
        )
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._login_lock = None  # created lazily so construction needs no loop

    async def aclose(self) -> None:
        await self._client.aclose()

    def _token_valid(self) -> bool:
        if not self._token or not self._token_expiry:
            return False
        return datetime.now(timezone.utc) < self._token_expiry - timedelta(seconds=60)

    async def _login(self) -> None:
        try:
            response = await self._client.post(
                "/api/auth/login",
                json={"email": self._settings.username,
                      "password": self._settings.password},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise NovaCronAPIError(f"NovaCron login failed: {exc}") from exc

        data = response.json()
        token = data.get("token")
        if not token:
            raise NovaCronAPIError("NovaCron login response did not contain a token")
        self._token = token
        self._token_expiry = _parse_rfc3339(data.get("expiresAt"))
        if self._token_expiry is None:
            # Server did not report an expiry; refresh conservatively.
            self._token_expiry = datetime.now(timezone.utc) + timedelta(minutes=25)

    async def _ensure_token(self) -> str:
        if self._token_valid():
            return self._token
        if self._login_lock is None:
            self._login_lock = asyncio.Lock()
        async with self._login_lock:
            if not self._token_valid():
                await self._login()
            return self._token

    async def _request(self, method: str, path: str) -> Any:
        attempts = 1 + max(0, self._settings.max_retries)
        retried_auth = False
        last_error: Optional[Exception] = None

        attempt = 0
        while attempt < attempts:
            try:
                token = await self._ensure_token()
                response = await self._client.request(
                    method, path,
                    headers={"Authorization": f"Bearer {token}"},
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
            else:
                if response.status_code == 401 and not retried_auth:
                    # Token rejected: drop it and re-login exactly once,
                    # without consuming a retry attempt.
                    retried_auth = True
                    self._token = None
                    self._token_expiry = None
                    continue
                if response.status_code >= 500:
                    last_error = NovaCronAPIError(
                        f"NovaCron {method} {path} returned {response.status_code}"
                    )
                else:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise NovaCronAPIError(
                            f"NovaCron {method} {path} failed: {exc}"
                        ) from exc
                    return response.json()
            attempt += 1
            if attempt < attempts:
                await asyncio.sleep(self._settings.retry_backoff * attempt)

        if last_error is not None:
            raise NovaCronAPIError(
                f"NovaCron {method} {path} unreachable after {attempts} attempts: "
                f"{last_error}"
            ) from last_error
        raise NovaCronAPIError(f"NovaCron {method} {path} failed after {attempts} attempts")

    async def get_host_metrics(self) -> Dict[str, Any]:
        """Host-level metrics for the node running the NovaCron API server."""
        data = await self._request("GET", "/api/v1/monitoring/metrics")
        return data if isinstance(data, dict) else {}

    async def get_vm_metrics(self) -> List[Dict[str, Any]]:
        """Per-VM metric samples (cpuUsage/memoryUsage present only when sampled)."""
        data = await self._request("GET", "/api/v1/monitoring/vms")
        return data if isinstance(data, list) else []


def fetch_metric_rows(host: Dict[str, Any], vms: List[Dict[str, Any]],
                      now: Optional[datetime] = None) -> Optional[List[Dict[str, Any]]]:
    """Map NovaCron monitoring payloads into model feature rows.

    Only keys backed by a measured sample are included. Returns ``None`` when
    no row carries any measurement, so callers back off instead of training or
    predicting on empty shells.
    """
    now = now or datetime.utcnow()
    timestamp = host.get("timestamp") if isinstance(host.get("timestamp"), str) else None
    timestamp = timestamp or now.isoformat()

    rows: List[Dict[str, Any]] = []

    host_row: Dict[str, Any] = {"node_id": "host", "timestamp": timestamp}
    measured = _measured(host_row, "cpu_usage", host.get("currentCpuUsage"))
    measured |= _measured(host_row, "memory_usage", host.get("currentMemoryUsage"))
    measured |= _measured(host_row, "disk_usage", host.get("currentDiskUsage"))
    measured |= _measured(host_row, "network_usage", host.get("currentNetworkUsage"))
    for key in ("loadAverage1m", "loadAverage5m", "loadAverage15m"):
        measured |= _measured(host_row, key, host.get(key))
    if measured:
        rows.append(host_row)

    for vm in vms:
        vm_id = vm.get("vmId")
        if not vm_id:
            continue
        row: Dict[str, Any] = {"node_id": vm_id, "timestamp": timestamp}
        if vm.get("name"):
            row["name"] = vm["name"]
        if vm.get("status"):
            row["status"] = vm["status"]
        measured = _measured(row, "cpu_usage", vm.get("cpuUsage"))
        measured |= _measured(row, "memory_usage", vm.get("memoryUsage"))
        if measured:
            rows.append(row)

    return rows or None


def fetch_resource_frame(host: Dict[str, Any]) -> Optional["pd.DataFrame"]:
    """Build the single-host resource-utilization frame for the optimizer.

    Only the host row is produced: it is the sole resource with measured cpu,
    memory, and storage utilization. All three must be present, so the
    frame never carries NaN utilization values into the optimizer's
    analysis. Returns ``None`` when any measurement is missing.
    """
    import pandas as pd

    measured = [
        host.get("currentCpuUsage"),
        host.get("currentMemoryUsage"),
        host.get("currentDiskUsage"),
    ]
    if any(not isinstance(v, (int, float)) for v in measured):
        return None

    host_id = host.get("hostname") if isinstance(host.get("hostname"), str) else "host"
    return pd.DataFrame([{
        "resource_id": host_id,
        "cpu_utilization": measured[0] / 100.0,
        "memory_utilization": measured[1] / 100.0,
        "storage_utilization": measured[2] / 100.0,
    }])


def _measured(row: Dict[str, Any], key: str, value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        row[key] = float(value)
        return True
    return False
