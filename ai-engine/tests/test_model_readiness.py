"""Readiness gate and on-disk model loading."""

import logging

import joblib
import numpy as np
import pandas as pd

from ai_engine.models.base import BaseMLModel, ModelMetadata, ModelType
from ai_engine.models.persistence import (
    PAYLOAD_TYPE_KEY,
    dump_typed_model,
    load_stored_models,
    load_typed_model,
    read_payload_type,
)
from ai_engine.models.persistence import PayloadTypeMismatch


class _StubModel(BaseMLModel):
    def train(self, X, y, validation_data=None):
        self.mark_trained(object())
        return {}

    def predict(self, X):
        return np.array([])

    def predict_proba(self, X):
        return np.array([])

    def save_model(self, filepath):
        return None

    def load_model(self, filepath):
        if filepath.endswith("bad.joblib"):
            raise ValueError("incompatible artifact")
        self.mark_trained(object())


def test_training_flag_alone_does_not_make_a_model_ready():
    model = _StubModel(ModelMetadata(
        model_id="stub",
        model_type=ModelType.FAILURE_PREDICTION,
        version="1.0.0",
    ))
    model._is_trained = True
    assert model.is_trained is False

    model.train(pd.DataFrame(), pd.Series(dtype=float))
    assert model.is_trained is True


def test_load_stored_models_keeps_compatible_artifacts_in_mtime_order(tmp_path):
    import os
    older = tmp_path / "older.joblib"
    newer = tmp_path / "newer.joblib"
    bad = tmp_path / "bad.joblib"
    dump_typed_model(str(older), "stub", {"n": 1})
    dump_typed_model(str(newer), "stub", {"n": 2})
    dump_typed_model(str(bad), "other_service", {"n": 3})
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(bad, (1_700_000_100, 1_700_000_100))
    os.utime(newer, (1_700_000_200, 1_700_000_200))

    loaded = load_stored_models(
        str(tmp_path),
        lambda model_id: _StubModel(ModelMetadata(
            model_id=model_id,
            model_type=ModelType.FAILURE_PREDICTION,
            version="1.0.0",
        )),
        "stub",
    )

    assert list(loaded) == ["older", "newer"]
    assert loaded["newer"].is_trained is True


def test_foreign_payload_is_not_unpickled(tmp_path, monkeypatch):
    foreign = tmp_path / "foreign.joblib"
    foreign.write_bytes(b"NOVACRON anomaly_detection\nNOT-A-PICKLE")
    constructed = []

    def refuse_unpickle(*_args, **_kwargs):
        raise AssertionError("foreign artifact was unpickled")

    monkeypatch.setattr(joblib, "load", refuse_unpickle)
    loaded = load_stored_models(
        str(tmp_path),
        lambda model_id: constructed.append(model_id),
        "failure_prediction",
    )

    assert loaded == {}
    assert constructed == []
    assert read_payload_type(foreign) == "anomaly_detection"


def test_corrupt_matching_artifact_logs_exception_type(tmp_path, caplog):
    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"NOVACRON failure_prediction\nNOT-A-PICKLE")

    class _LoadingModel(_StubModel):
        def load_model(self, filepath):
            load_typed_model(filepath, "failure_prediction")

    with caplog.at_level(logging.WARNING):
        loaded = load_stored_models(
            str(tmp_path),
            lambda model_id: _LoadingModel(ModelMetadata(
                model_id=model_id,
                model_type=ModelType.FAILURE_PREDICTION,
                version="1.0.0",
            )),
            "failure_prediction",
        )

    assert loaded == {}
    assert "broken.joblib" in caplog.text
    assert "KeyError:" in caplog.text


def test_typed_round_trip_and_inner_mismatch(tmp_path):
    path = tmp_path / "model.joblib"
    dump_typed_model(str(path), "failure_prediction", {"xgb_model": "estimator"})
    payload = load_typed_model(str(path), "failure_prediction")
    assert payload[PAYLOAD_TYPE_KEY] == "failure_prediction"
    assert payload["xgb_model"] == "estimator"

    buffer = __import__("io").BytesIO()
    joblib.dump({PAYLOAD_TYPE_KEY: "anomaly_detection"}, buffer)
    mismatched = tmp_path / "mislabeled.joblib"
    mismatched.write_bytes(b"NOVACRON failure_prediction\n" + buffer.getvalue())
    try:
        load_typed_model(str(mismatched), "failure_prediction")
    except PayloadTypeMismatch as exc:
        assert "anomaly_detection" in str(exc)
    else:
        raise AssertionError("inner payload_type mismatch was accepted")


def test_load_stored_models_missing_directory_is_empty(tmp_path):
    loaded = load_stored_models(str(tmp_path / "missing"), lambda model_id: None, "stub")
    assert loaded == {}
