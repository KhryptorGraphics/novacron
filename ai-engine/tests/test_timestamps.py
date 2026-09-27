"""Regression tests for the datetime.utcnow() -> timezone-aware UTC migration.

Guards the failure modes the migration must avoid: naive/aware mixing in a
store-then-compare pipeline, a double timezone suffix on serialized
timestamps, a changed wire shape on fields clients already consume, and a
local-time fallback next to UTC row timestamps.
"""

import json
import logging
from datetime import timedelta
from unittest.mock import patch

import pandas as pd

from ai_engine.core.anomaly_detector import AnomalyDetectionService
from ai_engine.core.resource_optimizer import ResourceOptimizationService, ResourceRecommendation
from ai_engine.integrations.novacron_client import _parse_rfc3339, fetch_metric_rows
from ai_engine.models.base import PredictionRequest, PredictionResponse
from ai_engine.utils.clock import utc_now, utc_now_iso_z, utc_now_naive
from ai_engine.utils.feature_engineering import AnomalyFeatureExtractor
from ai_engine.utils.logging import JSONFormatter
from ai_engine.utils.metrics import SystemMetricsCollector


def test_utc_now_is_timezone_aware():
    assert utc_now().tzinfo is not None
    assert utc_now().utcoffset() == timedelta(0)


def test_utc_now_naive_matches_the_same_instant():
    aware = utc_now()
    naive = utc_now_naive()
    assert naive.tzinfo is None
    assert abs((aware.replace(tzinfo=None) - naive).total_seconds()) < 1


def test_utc_now_iso_z_has_exactly_one_trailing_z():
    text = utc_now_iso_z()
    assert text.endswith("Z")
    assert "+00:00" not in text
    parsed = _parse_rfc3339(text)
    assert parsed is not None and parsed.tzinfo is not None


def test_json_formatter_emits_single_z_suffix_not_double():
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    payload = json.loads(JSONFormatter().format(record))
    ts = payload["timestamp"]
    assert ts.endswith("Z")
    assert "+00:00" not in ts
    assert _parse_rfc3339(ts) is not None


def test_prediction_response_timestamp_keeps_bare_iso_wire_shape():
    """PredictionResponse crosses /api/v1/*/predict and /detect verbatim;
    its timestamp must keep the pre-migration bare-ISO (no offset) shape."""
    response = PredictionResponse(
        request_id="r1", model_id="m1", prediction=0, response_time=0.01,
    )
    serialized = response.model_dump(mode="json")["timestamp"]
    assert "+00:00" not in serialized
    assert not serialized.endswith("Z")
    assert abs((utc_now_naive() - response.timestamp).total_seconds()) < 5


def test_prediction_request_timestamp_defaults_close_to_now():
    request = PredictionRequest(request_id="r1", features={"a": 1.0})
    assert request.timestamp.tzinfo is not None
    assert abs((utc_now() - request.timestamp).total_seconds()) < 5


async def test_anomaly_trends_exclude_stale_entries_without_raising(mock_anomaly_service):
    """get_anomaly_trends parses stored timestamps back with fromisoformat;
    a regression that makes the store and the cutoff disagree on awareness
    raises TypeError instead of filtering correctly."""
    service: AnomalyDetectionService = mock_anomaly_service
    request = PredictionRequest(request_id="r1", features={"cpu": 0.9})
    response = PredictionResponse(
        request_id="r1", model_id="m1", prediction=1, response_time=0.01, confidence=0.9,
    )
    await service._store_anomaly(
        request, response,
        {"anomaly_score": 0.9, "anomaly_types": ["performance"], "severity": "high"},
    )
    stale = dict(service._anomaly_history[-1])
    stale["timestamp"] = (utc_now() - timedelta(hours=2)).isoformat()
    service._anomaly_history.insert(0, stale)

    trends = await service.get_anomaly_trends(time_window=timedelta(hours=1))

    assert trends["total_anomalies"] == 1


async def test_optimization_summary_excludes_stale_entries_without_raising(mock_resource_service):
    service: ResourceOptimizationService = mock_resource_service
    recent = ResourceRecommendation(
        action="scale_down", target_resources={"cpu_cores": 2},
        expected_impact={"cost_change": -0.3}, confidence=0.8,
        reasoning=["low utilization"],
    )
    await service._store_recommendation(recent)

    stale = dict(service._recommendation_history[-1])
    stale["timestamp"] = (utc_now_naive() - timedelta(hours=2)).isoformat()
    service._recommendation_history.insert(0, stale)

    summary = await service.get_optimization_summary(time_window=timedelta(hours=1))

    assert summary["total_recommendations"] == 1


def test_resource_recommendation_timestamp_keeps_bare_iso_wire_shape():
    """ResourceRecommendation.to_dict() is returned verbatim by
    /api/v1/resource/optimize; the timestamp must stay offset-free."""
    rec = ResourceRecommendation(
        action="scale_up", target_resources={"cpu_cores": 4},
        expected_impact={}, confidence=0.5, reasoning=[],
    )
    serialized = rec.to_dict()["timestamp"]
    assert "+00:00" not in serialized
    assert not serialized.endswith("Z")


def test_system_metrics_collector_time_window_filter_does_not_mix_naive_aware():
    collector = SystemMetricsCollector()
    collector.collect_prediction_metrics("failure", "m1", 0.05, prediction=1, confidence=0.9)
    stale = dict(collector.metrics_history[-1])
    stale["timestamp"] = (utc_now() - timedelta(hours=2)).isoformat()
    collector.metrics_history.insert(0, stale)

    summary = collector.get_service_performance_summary("failure", time_window=timedelta(hours=1))

    assert summary["total_requests"] == 1


def test_fetch_metric_rows_timestamp_fallback_matches_rfc3339_z_shape():
    """When the host payload carries no timestamp, the generated fallback
    must use the same RFC3339 'Z' shape as host-provided timestamps, so a
    downstream consumer never sees a naive string next to an aware one."""
    rows = fetch_metric_rows(
        {"currentCpuUsage": 10.0, "currentMemoryUsage": 20.0,
         "currentDiskUsage": 30.0, "currentNetworkUsage": 1.0},
        [],
    )
    assert rows is not None
    timestamp = rows[0]["timestamp"]
    assert timestamp.endswith("Z")
    assert "+00:00" not in timestamp
    parsed = _parse_rfc3339(timestamp)
    assert parsed is not None
    assert abs((utc_now() - parsed).total_seconds()) < 5


def test_temporal_feature_fallback_uses_utc_not_local_time():
    """A row without a timestamp must get UTC hour_of_day, matching the
    RFC 3339 UTC timestamps every NovaCron-sourced row carries."""
    utc_instant = pd.Timestamp("2026-09-26T03:00:00Z")
    local_instant = utc_instant.tz_convert("Etc/GMT-9").tz_localize(None)  # 12:00 local

    def fake_now(tz=None):
        return utc_instant if tz is not None else local_instant

    with patch("ai_engine.utils.feature_engineering.pd.Timestamp.now", side_effect=fake_now):
        features = AnomalyFeatureExtractor()._extract_temporal_features(pd.Series({"cpu_utilization": 0.5}))

    assert features["hour_of_day"] == 3
    assert features["is_business_hours"] is False
