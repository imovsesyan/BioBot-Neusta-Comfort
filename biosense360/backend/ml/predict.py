"""
ML inference module for BioSense360.

Loads the trained model bundle from `model.pkl` (written by
`scripts/f9_uc9_comfort_class_classifier.py`) and exposes a single
`predict()` function used by the `/api/ml/info` and `/api/comfort-class`
routes.

If `model.pkl` does not exist, a rule-based fallback using
`humidex_service.classify_humidex()` is returned transparently.

Updated 2026-05-08: 4-class OHCOW (Extreme Danger merged into Danger,
n=3 in the real Météo France 2024-2025 dataset). Feature columns now
include `dew_point`, `wind_speed`, `rain_1h` plus their
`*_c / *_pct / *_mps / *_mm` variants for compatibility with the
training-script bundle.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from backend.services.humidex_service import classify_humidex, compute_humidex

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ML_DIR = Path(__file__).parent
MODEL_PATH = ML_DIR / "model.pkl"
RESULTS_PATH = ML_DIR / "benchmark_results.json"

# ---------------------------------------------------------------------------
# Integer → class label mapping (mirror of CLASS_ENCODING in the training
# script `scripts/f9_uc9_comfort_class_classifier.py`)
#   0: Comfortable        humidex < 30
#   1: Caution            30 <= humidex < 40
#   2: Extreme Caution    40 <= humidex < 45
#   3: Danger             humidex >= 45  (incl. Extreme Danger >= 54)
# ---------------------------------------------------------------------------
INT_TO_CLASS: dict[int, str] = {
    0: "Comfortable",
    1: "Caution",
    2: "Extreme Caution",
    3: "Danger",
}

# Default ordering of features expected by the model. Will be replaced by
# the value stored in the bundle if present.
DEFAULT_FEATURES: list[str] = [
    "temperature", "humidity", "dew_point",
    "wind_speed", "rain_1h",
    "hour", "month", "day_of_week",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]

# Module-level cache — model is loaded once per process lifetime
_bundle: Optional[dict] = None
_bundle_loaded: bool = False


def _classify_humidex_4class(humidex: float) -> str:
    """4-class OHCOW rule (Extreme Danger merged with Danger)."""
    if humidex < 30:
        return "Comfortable"
    if humidex < 40:
        return "Caution"
    if humidex < 45:
        return "Extreme Caution"
    return "Danger"


def _load_bundle() -> Optional[dict]:
    """
    Load the pickled model bundle from disk.

    The bundle written by the training script is a dict with keys:
        model, model_name, feature_columns, class_encoding,
        int_to_class, n_classes, n_samples_train,
        test_accuracy, test_f1_macro, leakage_audit.

    For backwards compatibility, if model.pkl contains a bare estimator
    (older format), it is wrapped on the fly.
    """
    global _bundle, _bundle_loaded
    if _bundle_loaded:
        return _bundle

    _bundle_loaded = True
    if not MODEL_PATH.exists():
        return None

    try:
        loaded = joblib.load(MODEL_PATH)
    except Exception:
        return None

    if isinstance(loaded, dict) and "model" in loaded:
        _bundle = loaded
    else:
        # Bare estimator → wrap with defaults
        _bundle = {
            "model": loaded,
            "model_name": type(loaded).__name__,
            "feature_columns": DEFAULT_FEATURES,
            "class_encoding": {v: k for k, v in INT_TO_CLASS.items()},
        }
    return _bundle


def predict(
    temperature: float,
    humidity: float,
    hour: int,
    month: int,
    lat: float,
    lon: float,
    is_indoor: bool,
) -> dict:
    """
    Predict the thermal comfort class for a single observation.

    Uses the trained ML model from `model.pkl` if available; otherwise
    falls back to the rule-based humidex classification (Masterton &
    Richardson 1979, OHCOW 4-class).

    The signature keeps `lat`/`lon`/`is_indoor` for API stability with
    the existing routers, even though the current model does not use
    them. Per-slot data not provided by the router (`dew_point`,
    `wind_speed`, `rain_1h`) is filled with sensible defaults:
      - dew_point: Magnus-formula approximation from T and RH
      - wind_speed: 0 m/s (calm)
      - rain_1h: 0 mm (dry)

    Args:
        temperature: Dry-bulb temperature in °C.
        humidity:    Relative humidity in %.
        hour:        Hour of day (0–23).
        month:       Month of year (1–12).
        lat:         Station latitude (currently unused by the model).
        lon:         Station longitude (currently unused by the model).
        is_indoor:   True for an indoor sensor, False for outdoor.

    Returns:
        dict with keys:
            predicted_class: one of the 4 OHCOW classes
                             (Comfortable, Caution, Extreme Caution, Danger)
            confidence:      float in [0, 1] (1.0 for the rule-based fallback)
            humidex:         float, computed Masterton 1979 humidex
            model_name:      str, identifier of the model used
    """
    humidex_val = compute_humidex(temperature, humidity)
    bundle = _load_bundle()

    # Approximate dew point from temperature and humidity when not measured
    # directly (simple Magnus-style approximation).
    dew_point_approx = float(temperature) - ((100.0 - float(humidity)) / 5.0)

    if bundle is None:
        cls = _classify_humidex_4class(humidex_val)
        return {
            "predicted_class": cls,
            "confidence": 1.0,
            "humidex": round(humidex_val, 2),
            "model_name": "rule_based_humidex",
        }

    model = bundle["model"]
    feature_columns: list[str] = bundle.get("feature_columns", DEFAULT_FEATURES)
    model_name: str = bundle.get("model_name", type(model).__name__)

    # Provide BOTH the training-script names (`*_c`, `*_pct`, `*_mps`,
    # `*_mm`) and the short names so this function works regardless of
    # which naming convention the bundle stores in `feature_columns`.
    feature_values: dict[str, float] = {
        # Training-script names
        "temperature_c":         float(temperature),
        "relative_humidity_pct": float(humidity),
        "dew_point_c":           dew_point_approx,
        "wind_speed_mps":        0.0,
        "rain_1h_mm":            0.0,
        # Short names (back-compat)
        "temperature":           float(temperature),
        "humidity":              float(humidity),
        "dew_point":             dew_point_approx,
        "wind_speed":            0.0,
        "rain_1h":               0.0,
        # Time features
        "hour":                  int(hour),
        "month":                 int(month),
        "day_of_week":           0,  # not derivable from hour/month alone
        "hour_sin":              math.sin(2 * math.pi * float(hour) / 24.0),
        "hour_cos":              math.cos(2 * math.pi * float(hour) / 24.0),
        "month_sin":             math.sin(2 * math.pi * float(month) / 12.0),
        "month_cos":             math.cos(2 * math.pi * float(month) / 12.0),
        # Tolerated for older bundles that included these
        "lat":                   float(lat),
        "lon":                   float(lon),
        "is_indoor":             int(bool(is_indoor)),
    }

    try:
        row = np.array([[feature_values[c] for c in feature_columns]], dtype=float)
        label_int = int(model.predict(row)[0])
        predicted_class = INT_TO_CLASS.get(
            label_int, _classify_humidex_4class(humidex_val)
        )

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(row)[0]
                confidence = float(np.max(proba))
            except Exception:
                confidence = 0.85
        else:
            confidence = 0.85

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "humidex": round(float(humidex_val), 2),
            "model_name": model_name,
        }
    except Exception:
        return {
            "predicted_class": _classify_humidex_4class(humidex_val),
            "confidence": 1.0,
            "humidex": round(float(humidex_val), 2),
            "model_name": "rule_based_humidex_fallback",
        }


def get_model_info() -> dict:
    """
    Return metadata about the current model for the GET /api/ml/info endpoint.

    Reads `benchmark_results.json` if available; otherwise returns a
    placeholder indicating that benchmarking has not yet been run.
    """
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                data = json.load(f)
            meta = data.get("meta", {})
            return {
                "model_available": MODEL_PATH.exists(),
                "model_path": str(MODEL_PATH),
                "selected_model": data.get("selected_model"),
                "accuracy": data.get("accuracy"),
                "f1_score": data.get("f1_score"),
                "inference_ms": data.get("inference_ms"),
                "top_features": data.get("top_features", []),
                "selection_reason": data.get("selection_reason", ""),
                "feature_columns": meta.get("feature_columns", DEFAULT_FEATURES),
                "class_encoding": meta.get("class_encoding", {}),
                "feature_importance": meta.get("feature_importance", {}),
                "all_results": data.get("all_results", []),
                "rejected_models": data.get("rejected_models", []),
                "leakage_audit": meta.get("leakage_audit", {}),
                "benchmark_run": True,
            }
        except Exception:
            pass

    return {
        "model_available": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "selected_model": "DecisionTree (placeholder — benchmark pending)",
        "accuracy": None,
        "f1_score": None,
        "inference_ms": None,
        "top_features": [],
        "feature_columns": DEFAULT_FEATURES,
        "class_encoding": {v: k for k, v in INT_TO_CLASS.items()},
        "feature_importance": {},
        "all_results": [],
        "rejected_models": [],
        "benchmark_run": False,
        "message": (
            "Benchmark has not been executed yet. "
            "Run `python scripts/f9_uc9_comfort_class_classifier.py` from the "
            "project root to train and evaluate all candidate models. Until "
            "then, the API uses rule-based 4-class Humidex classification."
        ),
    }
