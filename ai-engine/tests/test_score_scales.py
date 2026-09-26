"""One-row score scaling must stay finite and keep detector magnitude."""

import numpy as np
import pandas as pd

from ai_engine.core.anomaly_detector import AnomalyDetectionModel
from ai_engine.core.failure_predictor import FailurePredictionModel
from ai_engine.models.base import ModelMetadata, ModelType


def _failure_model() -> FailurePredictionModel:
    return FailurePredictionModel(ModelMetadata(
        model_id="failure-scale",
        model_type=ModelType.FAILURE_PREDICTION,
        version="1.0.0",
    ))


def _anomaly_model() -> AnomalyDetectionModel:
    return AnomalyDetectionModel(ModelMetadata(
        model_id="anomaly-scale",
        model_type=ModelType.ANOMALY_DETECTION,
        version="1.0.0",
    ))


def test_one_row_failure_score_uses_training_range():
    model = _failure_model()
    model._isolation_score_low = -0.2
    model._isolation_score_high = 0.2

    proba = model._scores_to_proba(np.array([-0.2]))

    assert np.isfinite(proba).all()
    assert proba.shape == (1, 2)
    assert proba[0, 1] == 1.0


def test_failure_score_without_range_is_not_nan():
    model = _failure_model()

    proba = model._scores_to_proba(np.array([-2.0]))

    assert np.isfinite(proba).all()
    assert proba[0, 1] > 0.5


def test_one_row_anomaly_score_uses_training_range():
    model = _anomaly_model()
    model._detector_score_range["isolation_forest"] = (-0.4, 0.4)

    scores = model._normalize_scores(np.array([-0.4]), "isolation_forest")

    assert scores.shape == (1,)
    assert scores[0] == 0.0


def test_constant_anomaly_score_is_not_forced_to_one_half():
    model = _anomaly_model()

    scores = model._normalize_scores(np.array([2.0]), "isolation_forest")

    assert np.isfinite(scores).all()
    assert scores[0] != 0.5


def test_isolation_forest_inlier_scores_below_outlier():
    """A healthy IsolationForest point must not outrank a real outlier."""
    model = _anomaly_model()
    model.mark_trained(object())
    model._feature_names = ["cpu"]
    model._detector_score_range["isolation_forest"] = (-0.4, 0.4)
    model._statistical_thresholds = {}

    class _IdentityExtractor:
        def extract_features(self, frame):
            return frame

    class _IdentityScaler:
        def transform(self, frame):
            return frame.to_numpy(dtype=float)

    class _Forest:
        def decision_function(self, _scaled):
            # sklearn: positive is an inlier, negative is an outlier.
            return np.array([0.4, -0.4])

    model._feature_extractor = _IdentityExtractor()
    model._scaler = _IdentityScaler()
    model._isolation_forest = _Forest()

    scores = model.decision_function(pd.DataFrame({"cpu": [0.2, 9.0]}))

    assert scores[0] < scores[1]
    assert scores[0] < 0.5
    assert scores[1] > scores[0]


def test_missing_optional_detectors_do_not_shrink_the_score():
    """A full outlier stays a full ensemble vote when AE and LSTM are absent."""
    model = _anomaly_model()
    model.mark_trained(object())
    model._feature_names = ["cpu"]
    model._detector_score_range["isolation_forest"] = (-0.4, 0.4)
    model._statistical_thresholds = {}

    class _IdentityExtractor:
        def extract_features(self, frame):
            return frame

    class _IdentityScaler:
        def transform(self, frame):
            return frame.to_numpy(dtype=float)

    class _Forest:
        def decision_function(self, _scaled):
            return np.array([-0.4])

    model._feature_extractor = _IdentityExtractor()
    model._scaler = _IdentityScaler()
    model._isolation_forest = _Forest()
    model._lof_detector = None
    model._ocsvm_detector = None
    model._autoencoder = None
    model._lstm_detector = None

    alone = model.decision_function(pd.DataFrame({"cpu": [9.0]}))

    class _Outlier:
        def decision_function(self, _scaled):
            return np.array([0.4])

    model._detector_score_range["lof"] = (-0.4, 0.4)
    model._lof_detector = _Outlier()
    with_lof = model.decision_function(pd.DataFrame({"cpu": [9.0]}))

    assert alone[0] == 0.8
    assert with_lof[0] == 0.8


def test_statistical_anomalies_use_scaled_feature_values():
    model = _anomaly_model()
    model._feature_names = ["cpu"]
    model._statistical_thresholds = {
        "cpu": {
            "mean": 0.0,
            "std": 1.0,
            "z_threshold": 3.0,
            "lower_bound": -1.5,
            "upper_bound": 1.5,
        }
    }

    in_range = model._detect_statistical_anomalies(pd.DataFrame({"cpu": [0.2]}))
    outside = model._detect_statistical_anomalies(pd.DataFrame({"cpu": [10.0]}))

    assert in_range[0] == 0.0
    assert outside[0] > 0.0
