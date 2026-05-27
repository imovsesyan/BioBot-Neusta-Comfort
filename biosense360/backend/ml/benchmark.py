"""
ML Benchmarking script for BioSense360.

Generates a synthetic, in-memory dataset of thermal comfort conditions
(no PostgreSQL connection required), trains a panel of classifier
candidates, reports accuracy / F1 / approximate inference time, saves
the best model to ml/model.pkl, and writes a benchmark summary to
ml/benchmark_results.json.

Candidates evaluated:
    - Logistic Regression                       (baseline, linear)
    - K-Nearest Neighbors                       (instance-based)
    - Support Vector Machine (RBF)              (kernel)
    - Decision Tree                             (single tree)
    - Random Forest                             (bagging)
    - XGBoost (or sklearn HistGradientBoosting fallback)
    - LightGBM (or sklearn GradientBoosting fallback)
    - MLP Neural Network                        (MLPClassifier)
    - Voting Ensemble  (top-3, soft voting)
    - Stacking Ensemble (top-3, LR meta-learner)

Features:
    temperature, humidity, humidex, hour, month, day_of_week, is_indoor

Label:
    comfort_class derived strictly from humidex via OHCOW 2022 thresholds
    (no Neusta vivabilite_binary_mean is used → no target leakage).

Run from the backend directory (no DB required):
    python ml/benchmark.py
or as a module:
    python -m backend.ml.benchmark
"""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from pathlib import Path

# Allow running as a top-level script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ML_DIR = Path(__file__).parent
MODEL_PATH = ML_DIR / "model.pkl"
RESULTS_PATH = ML_DIR / "benchmark_results.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SAMPLES = 8000
CV_FOLDS = 5

# Comfort class → integer encoding (ordered by severity).
# IMPORTANT: labels are derived from humidex thresholds (OHCOW 2022),
# never from any vivabilite_* target → no leakage.
CLASS_ENCODING: dict[str, int] = {
    "Comfortable":     0,
    "Caution":         1,
    "Extreme Caution": 2,
    "Danger":          3,
    "Extreme Danger":  4,
}
INT_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_ENCODING.items()}

# Features used by every model. Excludes lat/lon (geographic identifiers
# are not predictive of thermal comfort directly and would invite spatial
# leakage between train/test in a real deployment). Excludes any
# vivabilite_* fields by construction (target leakage policy).
FEATURE_COLUMNS: list[str] = [
    "temperature",
    "humidity",
    "humidex",
    "hour",
    "month",
    "day_of_week",
    "is_indoor",
]

# Explainability rating per model family. Used to break ties when two
# models score within 0.005 accuracy of each other.
EXPLAINABILITY: dict[str, str] = {
    "LogisticRegression": "High",
    "DecisionTree":       "High",
    "KNN":                "Medium",
    "RandomForest":       "Medium",
    "XGBoost":            "Medium",
    "LightGBM":           "Medium",
    "GradientBoosting":   "Medium",
    "HistGradientBoosting": "Medium",
    "SVM":                "Low",
    "MLPNeuralNet":       "Low",
    "VotingEnsemble":     "Low",
    "StackingEnsemble":   "Low",
}


# ---------------------------------------------------------------------------
# Humidex (Masterton & Richardson 1979) — duplicated locally so the script
# is self-contained and does not require importing the FastAPI service.
# ---------------------------------------------------------------------------
def compute_humidex(temp_c: float, rh: float) -> float:
    """Masterton & Richardson 1979 humidex."""
    e = 6.112 * (10 ** (7.5 * temp_c / (237.7 + temp_c))) * (rh / 100.0)
    return temp_c + 0.5555 * (e - 10.0)


def classify_humidex(humidex: float) -> str:
    """OHCOW 2022 comfort classification."""
    if humidex < 30.0:
        return "Comfortable"
    elif humidex < 40.0:
        return "Caution"
    elif humidex <= 45.0:
        return "Extreme Caution"
    elif humidex <= 54.0:
        return "Danger"
    else:
        return "Extreme Danger"


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Build a realistic synthetic dataset of thermal comfort observations.

    Sampling strategy:
      - Months 1–12 uniform (full year).
      - Hours 0–23 uniform.
      - is_indoor Bernoulli(0.4).
      - Temperature: sinusoidal daily curve with seasonal amplitude
        + Gaussian noise. Indoor temps shifted ~−4 °C (A/C), reduced
        diurnal swing.
      - Humidity: inverse curve with offset for indoor (~−10 %).
      - Sample broadly enough to cover all 5 comfort classes.

    The label `comfort_class` is derived ONLY from humidex
    (no leakage from any other ground-truth source).
    """
    rng = np.random.default_rng(seed)

    hour       = rng.integers(0, 24, n_samples)
    month      = rng.integers(1, 13, n_samples)
    day_of_week = rng.integers(0, 7, n_samples)
    is_indoor  = rng.binomial(1, 0.4, n_samples)

    # Seasonal midpoint (°C): cold winter → hot summer for Toulouse-like climate.
    season_mid = 14.0 + 12.0 * np.cos((month - 7) * np.pi / 6.0)  # peak in July
    # Diurnal swing: ±10 °C outdoor, ±3 °C indoor
    diurnal_amp = np.where(is_indoor == 1, 3.0, 10.0)
    diurnal = -np.cos((hour - 14) * np.pi / 12.0) * diurnal_amp
    indoor_shift = np.where(is_indoor == 1, -4.0, 0.0)
    temperature = season_mid + diurnal + indoor_shift + rng.normal(0, 1.5, n_samples)

    # Inject some heat-wave outliers so all 5 classes are represented
    n_heatwave = n_samples // 12
    heatwave_idx = rng.choice(n_samples, n_heatwave, replace=False)
    temperature[heatwave_idx] += rng.uniform(8, 18, n_heatwave)

    temperature = np.clip(temperature, -10.0, 50.0)

    # Humidity: inverse of temperature within plausible bounds
    base_humid = 80.0 - (temperature - 10.0) * 1.2 + rng.normal(0, 5.0, n_samples)
    indoor_humid_shift = np.where(is_indoor == 1, -10.0, 0.0)
    humidity = np.clip(base_humid + indoor_humid_shift, 10.0, 100.0)

    # Compute humidex per row (vectorised version of Masterton 1979)
    e = 6.112 * np.power(10.0, 7.5 * temperature / (237.7 + temperature)) * (humidity / 100.0)
    humidex = temperature + 0.5555 * (e - 10.0)

    # Derive label strictly from humidex (no leakage)
    comfort_class = np.array(
        [classify_humidex(float(h)) for h in humidex],
        dtype=object,
    )

    df = pd.DataFrame({
        "temperature":  temperature,
        "humidity":     humidity,
        "humidex":      humidex,
        "hour":         hour,
        "month":        month,
        "day_of_week":  day_of_week,
        "is_indoor":    is_indoor,
        "comfort_class": comfort_class,
    })
    return df


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    df["label"] = df["comfort_class"].map(CLASS_ENCODING)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return X, y


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------
def build_candidates() -> dict[str, object]:
    """Build the dict of name → estimator. Models needing scaling are wrapped."""
    candidates: dict[str, object] = {}

    candidates["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, multi_class="auto", random_state=RANDOM_SEED)),
    ])

    candidates["KNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ])

    candidates["SVM"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)),
    ])

    candidates["DecisionTree"] = DecisionTreeClassifier(
        max_depth=12, random_state=RANDOM_SEED,
    )

    candidates["RandomForest"] = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1,
    )

    # XGBoost — fall back to sklearn HistGradientBoosting if not installed
    try:
        from xgboost import XGBClassifier
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1,
            tree_method="hist", verbosity=0,
        )
    except ImportError:
        print("  [info] xgboost not installed — falling back to HistGradientBoosting.")
        candidates["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=200, random_state=RANDOM_SEED,
        )

    # LightGBM — fall back to sklearn GradientBoosting if not installed
    try:
        from lightgbm import LGBMClassifier
        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=-1,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )
    except ImportError:
        print("  [info] lightgbm not installed — falling back to GradientBoosting.")
        candidates["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=150, random_state=RANDOM_SEED,
        )

    candidates["MLPNeuralNet"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=RANDOM_SEED,
            early_stopping=True,
        )),
    ])

    return candidates


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def measure_inference_ms(model, X_sample: np.ndarray, n_repeats: int = 3) -> float:
    """Approximate inference time in milliseconds per sample."""
    n = max(len(X_sample), 1)
    timings = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict(X_sample)
        timings.append((time.perf_counter() - t0))
    avg_total_s = float(np.mean(timings))
    return round(1000.0 * avg_total_s / n, 4)


def evaluate_one(
    name: str,
    model,
    X_train, y_train, X_test, y_test,
    cv_splitter,
) -> dict:
    """Train on (X_train, y_train), evaluate on (X_test, y_test) + CV on train."""
    print(f"  [{name}]", end=" ", flush=True)
    t_fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = round(time.perf_counter() - t_fit_start, 3)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1w = float(f1_score(y_test, y_pred, average="weighted"))

    # k-fold CV on the training set (stratified) using accuracy
    try:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=cv_splitter,
            scoring="accuracy", n_jobs=1,
        )
        cv_mean = float(np.mean(cv_scores))
        cv_std  = float(np.std(cv_scores))
    except Exception as exc:
        print(f"(CV failed: {exc})", end=" ")
        cv_mean, cv_std = float("nan"), float("nan")

    inf_ms = measure_inference_ms(model, X_test[:200])

    print(
        f"acc={acc:.4f}  F1={f1w:.4f}  "
        f"CV={cv_mean:.4f}±{cv_std:.4f}  inf={inf_ms} ms/sample  "
        f"fit={fit_seconds}s"
    )

    return {
        "name": name,
        "accuracy": round(acc, 6),
        "f1_weighted": round(f1w, 6),
        "cv_accuracy_mean": round(cv_mean, 6) if not math.isnan(cv_mean) else None,
        "cv_accuracy_std":  round(cv_std,  6) if not math.isnan(cv_std)  else None,
        "inference_ms_per_sample": inf_ms,
        "fit_seconds": fit_seconds,
        "explainability": EXPLAINABILITY.get(name, "Medium"),
        "leakage_warning": acc > 0.95,
    }


def get_feature_importances(model) -> dict[str, float] | None:
    """Best-effort feature importance extraction (handles Pipelines)."""
    inner = model
    if hasattr(inner, "named_steps") and "clf" in inner.named_steps:
        inner = inner.named_steps["clf"]
    if hasattr(inner, "feature_importances_"):
        imps = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        coef = inner.coef_
        imps = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    if len(imps) != len(FEATURE_COLUMNS):
        return None
    return {feat: round(float(v), 6) for feat, v in zip(FEATURE_COLUMNS, imps)}


# ---------------------------------------------------------------------------
# Main benchmark routine
# ---------------------------------------------------------------------------
def run_benchmark() -> None:
    print("BioSense360 — ML Benchmark (synthetic, standalone)")
    print("=" * 60)

    # ---- 1. Synthetic dataset ----
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=N_SAMPLES, seed=RANDOM_SEED)

    label_counts = df["comfort_class"].value_counts().to_dict()
    print(f"  Rows: {len(df):,}  |  Class distribution: {label_counts}")

    # Drop classes with fewer than CV_FOLDS samples (stratified CV needs ≥k each)
    keep_classes = {k for k, v in label_counts.items() if v >= CV_FOLDS}
    if len(keep_classes) < len(label_counts):
        dropped = set(label_counts) - keep_classes
        print(f"  [warn] Dropping classes with < {CV_FOLDS} samples: {dropped}")
        df = df[df["comfort_class"].isin(keep_classes)].reset_index(drop=True)

    X, y = prepare_features(df)
    print(f"  Feature matrix: {X.shape}, "
          f"label dist (encoded): {dict(zip(*np.unique(y, return_counts=True)))}")

    # ---- 2. 80/20 stratified split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED,
    )
    cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # ---- 3. Train + evaluate all base candidates ----
    print("\nTraining base candidates...")
    candidates = build_candidates()
    results: list[dict] = []
    fitted: dict[str, object] = {}
    for name, est in candidates.items():
        try:
            row = evaluate_one(name, est, X_train, y_train, X_test, y_test, cv_splitter)
            results.append(row)
            fitted[name] = est
        except Exception as exc:
            print(f"FAILED: {exc}")

    # ---- 4. Build Voting + Stacking from the top-3 base models ----
    base_sorted = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    top3_names  = [r["name"] for r in base_sorted[:3]]
    print(f"\nTop-3 base models for ensembles: {top3_names}")

    top3_estimators = [(name, fitted[name]) for name in top3_names]

    # Voting (soft if all support predict_proba, else hard)
    can_soft = all(
        hasattr(getattr(fitted[name], "named_steps", {}).get("clf", fitted[name]),
                "predict_proba")
        or hasattr(fitted[name], "predict_proba")
        for name in top3_names
    )
    voting_clf = VotingClassifier(
        estimators=[(n, m) for n, m in top3_estimators],
        voting="soft" if can_soft else "hard",
        n_jobs=1,
    )
    print("\nTraining ensembles...")
    try:
        results.append(
            evaluate_one("VotingEnsemble", voting_clf,
                         X_train, y_train, X_test, y_test, cv_splitter)
        )
        fitted["VotingEnsemble"] = voting_clf
    except Exception as exc:
        print(f"  VotingEnsemble FAILED: {exc}")

    # Stacking with Logistic Regression meta-learner
    stacking_clf = StackingClassifier(
        estimators=[(n, m) for n, m in top3_estimators],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        cv=3, n_jobs=1, passthrough=False,
    )
    try:
        results.append(
            evaluate_one("StackingEnsemble", stacking_clf,
                         X_train, y_train, X_test, y_test, cv_splitter)
        )
        fitted["StackingEnsemble"] = stacking_clf
    except Exception as exc:
        print(f"  StackingEnsemble FAILED: {exc}")

    # ---- 5. Selection logic ----
    # Sort primarily by accuracy desc, then by f1, then by inference speed asc.
    results.sort(
        key=lambda r: (r["accuracy"], r["f1_weighted"], -r["inference_ms_per_sample"]),
        reverse=True,
    )

    # Selection: if the top model has acc > 0.999 we still pick it but flag it;
    # all models scoring within 0.005 of the top get a tiebreak by explainability.
    top_acc = results[0]["accuracy"]
    contenders = [r for r in results if top_acc - r["accuracy"] <= 0.005]
    explain_rank = {"High": 0, "Medium": 1, "Low": 2}
    contenders.sort(
        key=lambda r: (
            explain_rank.get(r["explainability"], 3),
            r["inference_ms_per_sample"],
        )
    )
    selected = contenders[0]
    selected_name = selected["name"]
    selected_model = fitted[selected_name]

    selection_reason_parts = [
        f"Top accuracy {top_acc:.4f} on the held-out 20% test set;",
        f"selected '{selected_name}' "
        f"(acc={selected['accuracy']:.4f}, "
        f"F1={selected['f1_weighted']:.4f}, "
        f"CV={selected['cv_accuracy_mean']}, "
        f"inference {selected['inference_ms_per_sample']} ms/sample, "
        f"explainability={selected['explainability']}).",
    ]
    if len(contenders) > 1:
        selection_reason_parts.append(
            f"Tiebreak among {[c['name'] for c in contenders]} by explainability "
            "then inference latency."
        )
    if selected["leakage_warning"]:
        selection_reason_parts.append(
            "WARNING: accuracy > 0.95 — expected because the label is a deterministic "
            "function of humidex (a feature). This is not target leakage in the "
            "scientific sense (no vivabilite_* used), but the comfort_class is a "
            "rule-based projection of humidex thresholds. Treat as upper bound."
        )

    selection_reason = " ".join(selection_reason_parts)

    # Top-3 features by importance for the selected model
    feat_imp = get_feature_importances(selected_model) or {}
    top_features = [
        f for f, _ in sorted(feat_imp.items(), key=lambda kv: kv[1], reverse=True)
    ][:3] or FEATURE_COLUMNS[:3]

    rejected_models = [
        {
            "name": r["name"],
            "reason": (
                f"Lower accuracy ({r['accuracy']:.4f} vs selected {selected['accuracy']:.4f})"
                if r["accuracy"] < selected["accuracy"]
                else f"Within tie band but lower explainability "
                     f"({r['explainability']}) or slower "
                     f"({r['inference_ms_per_sample']} ms)"
            ),
        }
        for r in results if r["name"] != selected_name
    ]

    # ---- 6. Persist artefacts ----
    print(f"\nSelected model: {selected_name}")
    print(f"Saving model -> {MODEL_PATH}")
    joblib.dump(
        {
            "model": selected_model,
            "model_name": selected_name,
            "feature_columns": FEATURE_COLUMNS,
            "class_encoding": CLASS_ENCODING,
        },
        MODEL_PATH,
    )

    benchmark_json = {
        "selected_model": selected_name,
        "accuracy": round(selected["accuracy"], 4),
        "f1_score": round(selected["f1_weighted"], 4),
        "inference_ms": selected["inference_ms_per_sample"],
        "top_features": top_features,
        "selection_reason": selection_reason,
        "all_results": results,
        "rejected_models": rejected_models,
        # Extra metadata (does not break the required schema)
        "meta": {
            "n_samples": int(len(df)),
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
            "class_encoding": CLASS_ENCODING,
            "feature_importance": feat_imp,
            "cv_folds": CV_FOLDS,
            "split": "stratified 80/20",
            "random_seed": RANDOM_SEED,
            "leakage_audit": {
                "vivabilite_used": False,
                "label_source": "humidex thresholds (OHCOW 2022) — rule-based",
                "note": (
                    "Label is a deterministic function of the 'humidex' feature; "
                    "high accuracy is therefore expected. No Neusta vivabilite_* "
                    "fields are present in features or labels."
                ),
            },
        },
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(benchmark_json, f, indent=2, default=str)

    print(f"Benchmark results -> {RESULTS_PATH}")
    print("=" * 60)
    print("Benchmark complete.")


if __name__ == "__main__":
    run_benchmark()
