"""Shared modeling helpers for the clinical trial outcome notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


DATA_PATH = Path("data/model_ready.parquet")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = ARTIFACT_DIR / "models"
EXPLAIN_DIR = ARTIFACT_DIR / "explainability"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
FEATURE_COLUMNS_PATH = ARTIFACT_DIR / "feature_columns.json"

LABEL_NAMES = {0: "COMPLETED", 1: "TERMINATED", 2: "WITHDRAWN"}
SPLIT_YEAR = 2019
NON_FEATURE = {"label", "overall_status", "start_year"}
LEAKAGE_PRONE_COLS = {
    "trial_duration_days",
    "enrollment_actual",
    "log_enrollment",
}
STRICT_DEPLOYMENT_EXCLUDE_PREFIXES = (
    "marketing_status_",
    "review_priority_",
    "submission_type_",
)
STRICT_DEPLOYMENT_EXCLUDE_COLS = {
    "approval_year",
    "years_since_approval",
    "priority_review",
}
RANDOM_STATE = 42


def ensure_artifact_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)


def load_model_ready() -> tuple[pl.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run the preprocessing notebooks first.")

    df = pl.read_parquet(DATA_PATH)
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE and df.schema[c].is_numeric()
    ]
    return df, feature_cols


def save_feature_columns(feature_cols: list[str]) -> None:
    FEATURE_COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")


def temporal_split(df: pl.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, ...]:
    train = df.filter(pl.col("start_year") < SPLIT_YEAR)
    test = df.filter(pl.col("start_year") >= SPLIT_YEAR)

    x_train = train.select(feature_cols).to_numpy().astype(np.float32)
    x_test = test.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    return x_train, x_test, y_train, y_test


def remove_leakage_prone_features(feature_cols: list[str]) -> list[str]:
    """Drop features not plausibly known near trial start."""
    return [c for c in feature_cols if c not in LEAKAGE_PRONE_COLS]


def remove_strict_deployment_risk_features(feature_cols: list[str]) -> list[str]:
    """Drop fields that require time-aware FDA snapshots for strict deployment claims."""
    base = remove_leakage_prone_features(feature_cols)
    return [
        c for c in base
        if c not in STRICT_DEPLOYMENT_EXCLUDE_COLS
        and not c.startswith(STRICT_DEPLOYMENT_EXCLUDE_PREFIXES)
    ]


def evaluate_model(
    model: Any,
    name: str,
    slug: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    training_seconds: float,
) -> dict[str, Any]:
    y_pred = model.predict(x_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=sorted(LABEL_NAMES))
    return {
        "name": name,
        "slug": slug,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "num_features": int(x_train.shape[1]),
        "split_year": SPLIT_YEAR,
        "training_seconds": round(training_seconds, 2),
    }


def save_model(model: Any, slug: str) -> Path:
    ensure_artifact_dirs()
    path = MODEL_DIR / f"{slug}.joblib"
    joblib.dump(model, path)
    return path


def save_or_update_metrics(metrics: dict[str, Any]) -> None:
    ensure_artifact_dirs()
    if METRICS_PATH.exists():
        payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        payload = {
            "label_names": LABEL_NAMES,
            "data_path": str(DATA_PATH),
            "split_year": SPLIT_YEAR,
            "models": [],
        }

    payload["models"] = [m for m in payload.get("models", []) if m.get("slug") != metrics["slug"]]
    payload["models"].append(metrics)
    payload["models"] = sorted(payload["models"], key=lambda m: m["name"])
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_saved_model(slug: str) -> Any:
    return joblib.load(MODEL_DIR / f"{slug}.joblib")


def validate_model_feature_count(slug: str, model: Any, feature_cols: list[str]) -> None:
    expected = getattr(model, "n_features_in_", None)
    if expected is None and hasattr(model, "feature_importances_"):
        expected = len(model.feature_importances_)

    if expected is not None and expected != len(feature_cols):
        raise ValueError(
            f"{slug} was trained with {expected} features, but the current feature list has "
            f"{len(feature_cols)}. Re-run Random_Forest_Model.ipynb and XGBoost_Model.ipynb "
            "before running Explainability_Analysis.ipynb."
        )


def write_tree_importance(slug: str, model: Any, feature_cols: list[str]) -> Path | None:
    if not hasattr(model, "feature_importances_"):
        return None

    validate_model_feature_count(slug, model, feature_cols)
    ensure_artifact_dirs()
    path = EXPLAIN_DIR / f"{slug}_feature_importance.csv"
    pl.DataFrame(
        {
            "feature": feature_cols,
            "importance": [float(v) for v in model.feature_importances_],
        }
    ).sort("importance", descending=True).write_csv(path)
    return path


def write_permutation_importance(
    slug: str,
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    sample_size: int = 1500,
) -> Path:
    validate_model_feature_count(slug, model, feature_cols)
    ensure_artifact_dirs()
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(sample_size, len(y_test))
    sample_idx = rng.choice(len(y_test), size=n, replace=False)
    result = permutation_importance(
        model,
        x_test[sample_idx],
        y_test[sample_idx],
        scoring="f1_macro",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    path = EXPLAIN_DIR / f"{slug}_permutation_importance.csv"
    pl.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": [float(v) for v in result.importances_mean],
            "importance_std": [float(v) for v in result.importances_std],
        }
    ).sort("importance_mean", descending=True).write_csv(path)
    return path


def write_shap_summary(
    slug: str,
    model: Any,
    x_test: np.ndarray,
    feature_cols: list[str],
    sample_size: int = 500,
) -> Path:
    import shap

    validate_model_feature_count(slug, model, feature_cols)
    ensure_artifact_dirs()
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(sample_size, x_test.shape[0])
    sample_idx = rng.choice(x_test.shape[0], size=n, replace=False)
    x_sample = x_test[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(values).mean(axis=0) for values in shap_values], axis=0)
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            mean_abs = np.abs(values).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(values).mean(axis=0)

    path = EXPLAIN_DIR / f"{slug}_shap_summary.csv"
    pl.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": [float(v) for v in mean_abs],
        }
    ).sort("mean_abs_shap", descending=True).write_csv(path)
    return path
