"""Tune XGBoost hyperparameters with grid, random, and Bayesian search."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from modeling_utils import (
    ARTIFACT_DIR,
    RANDOM_STATE,
    VALIDATION_START_YEAR,
    evaluate_model,
    load_model_ready,
    remove_leakage_prone_features,
    save_model,
    save_or_update_metrics,
    temporal_train_validation_test_split,
)


RESULTS_PATH = ARTIFACT_DIR / "xgboost_tuning_results.csv"
SUMMARY_PATH = ARTIFACT_DIR / "xgboost_tuning_summary.json"

BASE_XGB_PARAMS: dict[str, Any] = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

GRID_SPACE: dict[str, list[Any]] = {
    "n_estimators": [250, 400],
    "max_depth": [2, 4],
    "learning_rate": [0.03, 0.06],
    "subsample": [0.85],
    "colsample_bytree": [0.85],
    "reg_lambda": [1.0, 2.0],
}

RANDOM_SPACE: dict[str, list[Any]] = {
    "n_estimators": [150, 250, 400, 600],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.12],
    "subsample": [0.70, 0.85, 1.0],
    "colsample_bytree": [0.70, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.25, 0.75, 1.5],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

PARAM_COLUMNS = sorted(set(GRID_SPACE) | set(RANDOM_SPACE))


def make_xgb(params: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(**BASE_XGB_PARAMS, **params)


def fit_and_score(
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[float, float]:
    sample_weight = compute_sample_weight("balanced", y_train)
    model = make_xgb(params)

    start = time.perf_counter()
    model.fit(x_train, y_train, sample_weight=sample_weight)
    training_seconds = time.perf_counter() - start

    y_pred = model.predict(x_validation)
    score = f1_score(y_validation, y_pred, average="macro")
    return float(score), round(training_seconds, 2)


def result_row(
    method: str,
    trial_number: int,
    params: dict[str, Any],
    validation_macro_f1: float,
    training_seconds: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "trial_number": trial_number,
        "validation_macro_f1": validation_macro_f1,
        "training_seconds": training_seconds,
    }
    for key in PARAM_COLUMNS:
        row[key] = params.get(key)
    return row


def run_parameter_candidates(
    method: str,
    candidates: Iterable[dict[str, Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for trial_number, params in enumerate(candidates, start=1):
        score, training_seconds = fit_and_score(
            params,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        rows.append(result_row(method, trial_number, params, score, training_seconds))
        print(
            f"{method} trial {trial_number}: "
            f"validation_macro_f1={score:.4f}, params={params}"
        )
    return rows


def run_grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    max_candidates: int | None,
) -> list[dict[str, Any]]:
    candidates = list(ParameterGrid(GRID_SPACE))
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    return run_parameter_candidates(
        "grid",
        candidates,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )


def run_random_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    n_iter: int,
) -> list[dict[str, Any]]:
    candidates = ParameterSampler(
        RANDOM_SPACE,
        n_iter=n_iter,
        random_state=RANDOM_STATE,
    )
    return run_parameter_candidates(
        "random",
        candidates,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )


def run_bayesian_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    n_trials: int,
) -> list[dict[str, Any]]:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Bayesian tuning requires optuna. Install dependencies with "
            "`pip install -r requirements.txt`, or run with "
            "`--methods grid random`."
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: Any) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 8.0, log=True),
        }
        score, training_seconds = fit_and_score(
            params,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        trial.set_user_attr("training_seconds", training_seconds)
        print(
            f"bayesian trial {trial.number + 1}: "
            f"validation_macro_f1={score:.4f}, params={params}"
        )
        return score

    study.enqueue_trial(
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_lambda": 1.5,
        }
    )
    study.optimize(objective, n_trials=n_trials)

    rows = []
    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    for trial in completed_trials:
        rows.append(
            result_row(
                "bayesian",
                trial.number + 1,
                trial.params,
                float(trial.value),
                float(trial.user_attrs.get("training_seconds", 0.0)),
            )
        )
    return rows


def params_from_row(row: dict[str, Any]) -> dict[str, Any]:
    params = {}
    for key in PARAM_COLUMNS:
        value = row.get(key)
        if pd.notna(value):
            params[key] = value
    if "n_estimators" in params:
        params["n_estimators"] = int(params["n_estimators"])
    if "max_depth" in params:
        params["max_depth"] = int(params["max_depth"])
    if "min_child_weight" in params:
        params["min_child_weight"] = int(params["min_child_weight"])
    return params


def write_results(rows: list[dict[str, Any]], path: Path = RESULTS_PATH) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "method"],
        ascending=[False, True],
    ).to_csv(path, index=False)


def write_summary(summary: dict[str, Any], path: Path = SUMMARY_PATH) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["grid", "random", "bayesian"],
        default=["grid", "random", "bayesian"],
        help="Tuning methods to run.",
    )
    parser.add_argument(
        "--validation-start-year",
        type=int,
        default=VALIDATION_START_YEAR,
        help="First pre-test year used for validation.",
    )
    parser.add_argument(
        "--random-iterations",
        type=int,
        default=12,
        help="Number of random-search candidates.",
    )
    parser.add_argument(
        "--bayesian-trials",
        type=int,
        default=20,
        help="Number of Optuna/TPE Bayesian trials.",
    )
    parser.add_argument(
        "--max-grid-candidates",
        type=int,
        default=None,
        help="Optional cap for grid candidates, mainly for smoke tests.",
    )
    parser.add_argument(
        "--skip-refit",
        action="store_true",
        help="Do not train and save the final best model on train+validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df, feature_cols = load_model_ready()
    clean_feature_cols = remove_leakage_prone_features(feature_cols)
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        temporal_train_validation_test_split(
            df,
            clean_feature_cols,
            validation_start_year=args.validation_start_year,
        )
    )

    print(f"Clean feature count: {len(clean_feature_cols):,}")
    print(f"Tune train rows: {x_train.shape[0]:,}")
    print(f"Validation rows: {x_validation.shape[0]:,}")
    print(f"Final test rows: {x_test.shape[0]:,}")

    rows: list[dict[str, Any]] = []
    if "grid" in args.methods:
        rows.extend(
            run_grid_search(
                x_train,
                y_train,
                x_validation,
                y_validation,
                args.max_grid_candidates,
            )
        )
    if "random" in args.methods:
        rows.extend(
            run_random_search(
                x_train,
                y_train,
                x_validation,
                y_validation,
                args.random_iterations,
            )
        )
    if "bayesian" in args.methods:
        rows.extend(
            run_bayesian_search(
                x_train,
                y_train,
                x_validation,
                y_validation,
                args.bayesian_trials,
            )
        )

    if not rows:
        raise RuntimeError("No tuning results were produced.")

    write_results(rows)
    best_row = max(rows, key=lambda row: row["validation_macro_f1"])
    best_params = params_from_row(best_row)

    summary: dict[str, Any] = {
        "selection_metric": "validation_macro_f1",
        "validation_start_year": args.validation_start_year,
        "test_start_year": 2019,
        "num_features": len(clean_feature_cols),
        "methods": args.methods,
        "best_method": best_row["method"],
        "best_validation_macro_f1": best_row["validation_macro_f1"],
        "best_params": best_params,
        "results_path": str(RESULTS_PATH),
    }

    if not args.skip_refit:
        x_final_train = np.vstack([x_train, x_validation])
        y_final_train = np.concatenate([y_train, y_validation])
        sample_weight = compute_sample_weight("balanced", y_final_train)
        final_model = make_xgb(best_params)

        start = time.perf_counter()
        final_model.fit(x_final_train, y_final_train, sample_weight=sample_weight)
        training_seconds = time.perf_counter() - start

        metrics = evaluate_model(
            model=final_model,
            name="XGBoost Tuned",
            slug="xgboost_tuned",
            x_train=x_final_train,
            x_test=x_test,
            y_test=y_test,
            training_seconds=training_seconds,
        )
        metrics["tuning"] = {
            "selection_metric": "validation_macro_f1",
            "validation_start_year": args.validation_start_year,
            "best_method": best_row["method"],
            "best_validation_macro_f1": best_row["validation_macro_f1"],
            "best_params": best_params,
        }
        save_model(final_model, "xgboost_tuned")
        save_or_update_metrics(metrics)
        summary["final_test_macro_f1"] = metrics["macro_f1"]
        summary["final_test_accuracy"] = metrics["accuracy"]
        summary["model_path"] = "artifacts/models/xgboost_tuned.joblib"

    write_summary(summary)

    print("\nBest tuning result")
    print(f"Method: {best_row['method']}")
    print(f"Validation macro F1: {best_row['validation_macro_f1']:.4f}")
    print(f"Params: {best_params}")
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
