"""Tests for the shared NovaCron API client and metric row mapping.

These gate the monitoring-loop fix: the client must cache tokens, re-login
exactly once on a rejected token, and the mappers must omit unmeasured
metrics rather than fabricate them.
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from ai_engine.config import NovaCronSettings, SecuritySettings, Settings
from ai_engine.core.failure_predictor import FailurePredictionService
from ai_engine.core.resource_optimizer import ResourceOptimizationService
from ai_engine.integrations.novacron_client import (
    NovaCronAPIError,
    NovaCronClient,
    fetch_metric_rows,
    fetch_resource_frame,
)


def _settings(**kwargs) -> NovaCronSettings:
    defaults = dict(api_url="http://novacron.test", api_timeout=5,
                    max_retries=0, retry_backoff=0.0,
                    username="ai-engine", password="s3cret",
                    jwt_secret="test-secret")
    defaults.update(kwargs)
    return NovaCronSettings(**defaults)


def _expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=25)).isoformat()


class TestAuthAndRetry:
    async def test_token_is_cached_across_requests(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path == "/api/auth/login":
                return httpx.Response(200, json={"token": "tok-1", "expiresAt": _expiry()})
            assert request.headers["Authorization"] == "Bearer tok-1"
            return httpx.Response(200, json={"currentCpuUsage": 12.5})

        client = NovaCronClient(_settings(), transport=httpx.MockTransport(handler))
        try:
            first = await client.get_host_metrics()
            second = await client.get_host_metrics()
        finally:
            await client.aclose()

        assert first["currentCpuUsage"] == 12.5
        assert second["currentCpuUsage"] == 12.5
        assert calls.count("/api/auth/login") == 1

    async def test_rejected_token_triggers_exactly_one_relogin(self):
        logins = 0
        metrics_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal logins, metrics_calls
            if request.url.path == "/api/auth/login":
                logins += 1
                return httpx.Response(200, json={"token": f"tok-{logins}",
                                                 "expiresAt": _expiry()})
            metrics_calls += 1
            if request.headers["Authorization"] == "Bearer tok-1":
                return httpx.Response(401, json={"error": "expired"})
            return httpx.Response(200, json={"ok": True})

        client = NovaCronClient(_settings(), transport=httpx.MockTransport(handler))
        try:
            result = await client.get_host_metrics()
        finally:
            await client.aclose()

        assert result == {"ok": True}
        assert logins == 2
        assert metrics_calls == 2

    async def test_unreachable_api_raises_after_retries(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = NovaCronClient(_settings(max_retries=1),
                                transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(NovaCronAPIError):
                await client.get_host_metrics()
        finally:
            await client.aclose()

    async def test_login_failure_raises_novacron_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "invalid"})

        client = NovaCronClient(_settings(), transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(NovaCronAPIError):
                await client.get_host_metrics()
        finally:
            await client.aclose()


class TestMetricRowMapping:
    def test_unsampled_vm_row_is_dropped(self):
        rows = fetch_metric_rows(
            {"timestamp": "2026-09-26T00:00:00Z",
             "currentCpuUsage": 50.0, "currentMemoryUsage": 60.0,
             "currentDiskUsage": 70.0, "currentNetworkUsage": 1.5},
            [{"vmId": "vm-no-samples", "name": "idle", "status": "running"},
             {"vmId": "vm-sampled", "cpuUsage": 33.0}],
        )
        assert rows is not None
        node_ids = {row["node_id"] for row in rows}
        assert node_ids == {"host", "vm-sampled"}

        host_row = next(r for r in rows if r["node_id"] == "host")
        assert host_row["disk_usage"] == 70.0
        vm_row = next(r for r in rows if r["node_id"] == "vm-sampled")
        assert vm_row["cpu_usage"] == 33.0
        assert "memory_usage" not in vm_row  # unmeasured, not fabricated

    def test_empty_payload_returns_none(self):
        assert fetch_metric_rows({}, []) is None

    def test_resource_frame_requires_all_three_utilizations(self):
        assert fetch_resource_frame({"currentCpuUsage": 50.0,
                                     "currentMemoryUsage": 60.0}) is None

        frame = fetch_resource_frame({"currentCpuUsage": 50.0,
                                      "currentMemoryUsage": 60.0,
                                      "currentDiskUsage": 20.0})
        assert frame is not None
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["cpu_utilization"] == 0.5
        assert row["memory_utilization"] == 0.6
        assert row["storage_utilization"] == 0.2


class TestServiceFetchWiring:
    async def test_failure_service_fetch_uses_client(self, mock_settings):
        service = FailurePredictionService(mock_settings)

        class FakeClient:
            async def get_host_metrics(self):
                return {"currentCpuUsage": 10.0}

            async def get_vm_metrics(self):
                return []

        service._novacron_client = FakeClient()
        rows = await service._fetch_current_metrics()
        assert rows is not None
        assert rows[0]["node_id"] == "host"

    async def test_failure_service_fetch_returns_none_on_api_error(self, mock_settings):
        service = FailurePredictionService(mock_settings)

        class FailingClient:
            async def get_host_metrics(self):
                raise NovaCronAPIError("down")

            async def get_vm_metrics(self):
                raise NovaCronAPIError("down")

        service._novacron_client = FailingClient()
        assert await service._fetch_current_metrics() is None

    async def test_resource_service_fetch_builds_host_frame(self, mock_settings):
        service = ResourceOptimizationService(mock_settings)

        class FakeClient:
            async def get_host_metrics(self):
                return {"currentCpuUsage": 80.0, "currentMemoryUsage": 40.0,
                        "currentDiskUsage": 10.0}

        service._novacron_client = FakeClient()
        frame = await service._fetch_current_resource_states()
        assert frame is not None
        assert frame.iloc[0]["cpu_utilization"] == 0.8


class TestProductionSecretGuards:
    def test_placeholder_credentials_rejected_in_production(self):
        with pytest.raises(ValueError, match="NOVACRON_PASSWORD"):
            Settings(environment="production",
                     security=SecuritySettings(secret_key="real-secret"))

    def test_missing_jwt_secret_rejected_in_production(self):
        with pytest.raises(ValueError, match="NOVACRON_JWT_SECRET"):
            Settings(environment="production",
                     security=SecuritySettings(secret_key="real-secret"),
                     novacron=NovaCronSettings(password="real-password"))

    def test_real_credentials_accepted_in_production(self):
        settings = Settings(
            environment="production",
            security=SecuritySettings(secret_key="real-secret"),
            novacron=NovaCronSettings(password="real-password",
                                      jwt_secret="real-jwt"),
        )
        assert settings.environment == "production"

    def test_placeholders_allowed_outside_production(self):
        settings = Settings(environment="development")
        assert settings.novacron.password == "changeme"
