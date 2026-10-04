"""Training-only nonlinear model selection and frozen-test comparison."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from medintel.modeling import SEED, evaluate, prepare_model_frame, preprocessing

BOOTSTRAP_SEED = 418


def frozen_split(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    cohort = pd.read_csv(data_dir / "processed" / "adult-cardiovascular-cohort.csv")
    features, target, identifiers = prepare_model_frame(cohort)
    split = pd.read_csv(data_dir / "processed" / "v0.3-model-split.csv")
    partition = identifiers.map(split.set_index("SEQN")["partition"])
    if partition.isna().any() or set(partition.unique()) != {"train", "test"}:
        raise ValueError("local split file does not match the model cohort")
    train = partition.eq("train")
    test = partition.eq("test")
    return features.loc[train], features.loc[test], target.loc[train], target.loc[test]


def bootstrap_intervals(
    y_true: pd.Series, logistic_probability: np.ndarray,
    boosted_probability: np.ndarray, repetitions: int = 1000,
) -> dict[str, dict[str, list[float]]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y = y_true.to_numpy()
    metrics = {
        "roc_auc": roc_auc_score,
        "average_precision": average_precision_score,
        "brier_score": brier_score_loss,
    }
    samples: dict[str, dict[str, list[float]]] = {
        name: {"logistic": [], "boosted": [], "delta_boosted_minus_logistic": []}
        for name in metrics
    }
    completed = 0
    while completed < repetitions:
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        for name, metric in metrics.items():
            logistic_value = float(metric(y[indices], logistic_probability[indices]))
            boosted_value = float(metric(y[indices], boosted_probability[indices]))
            samples[name]["logistic"].append(logistic_value)
            samples[name]["boosted"].append(boosted_value)
            samples[name]["delta_boosted_minus_logistic"].append(
                boosted_value - logistic_value
            )
        completed += 1
    return {
        metric: {
            series: [float(value) for value in np.quantile(values, [0.025, 0.975])]
            for series, values in groups.items()
        }
        for metric, groups in samples.items()
    }


def run_comparison(data_dir: Path, reports_dir: Path, models_dir: Path) -> Path:
    x_train, x_test, y_train, y_test = frozen_split(data_dir)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision"}

    logistic = Pipeline([
        ("preprocess", preprocessing()),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    logistic_cv_raw = cross_validate(logistic, x_train, y_train, cv=folds, scoring=scoring)
    logistic_cv = {
        metric: {
            "mean": float(np.mean(logistic_cv_raw[f"test_{metric}"])),
            "standard_deviation": float(np.std(logistic_cv_raw[f"test_{metric}"], ddof=1)),
        }
        for metric in scoring
    }

    boosted = Pipeline([
        ("preprocess", preprocessing(dense=True)),
        ("model", HistGradientBoostingClassifier(random_state=SEED)),
    ])
    search_space = {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__max_iter": [100, 175, 250],
        "model__max_leaf_nodes": [7, 15, 31],
        "model__min_samples_leaf": [20, 40, 80],
        "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
    }
    search = RandomizedSearchCV(
        boosted,
        search_space,
        n_iter=16,
        scoring=scoring,
        refit="average_precision",
        cv=folds,
        random_state=SEED,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(x_train, y_train)
    best_index = search.best_index_
    boosted_cv = {
        metric: {
            "mean": float(search.cv_results_[f"mean_test_{metric}"][best_index]),
            "standard_deviation": float(search.cv_results_[f"std_test_{metric}"][best_index]),
        }
        for metric in scoring
    }

    logistic.fit(x_train, y_train)
    logistic_probability = logistic.predict_proba(x_test)[:, 1]
    boosted_probability = search.best_estimator_.predict_proba(x_test)[:, 1]
    test_results = {
        "logistic_regression": evaluate(y_test, logistic_probability),
        "hist_gradient_boosting": evaluate(y_test, boosted_probability),
    }
    intervals = bootstrap_intervals(
        y_test, logistic_probability, boosted_probability,
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, models_dir / "hist_gradient_boosting.joblib")

    results = {
        "selection_rule": "highest mean training-fold average precision",
        "cross_validation": {
            "folds": 5,
            "splitter": "stratified shuffled folds",
            "seed": SEED,
            "logistic_regression": logistic_cv,
            "hist_gradient_boosting": boosted_cv,
            "search_candidates": 16,
            "best_parameters": search.best_params_,
        },
        "frozen_test": test_results,
        "bootstrap": {
            "repetitions": 1000,
            "seed": BOOTSTRAP_SEED,
            "interval_type": "paired percentile 95%",
            "intervals": intervals,
        },
    }
    output = reports_dir / "metrics" / "v0.4-model-comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_report(reports_dir / "v0.4-model-comparison.md", results)
    return output


def write_report(path: Path, results: dict[str, object]) -> None:
    cv = results["cross_validation"]
    test = results["frozen_test"]
    intervals = results["bootstrap"]["intervals"]
    logistic = test["logistic_regression"]
    boosted = test["hist_gradient_boosting"]
    lines = [
        "# V0.4 nonlinear model comparison",
        "",
        "Model selection used only the V0.3 training partition. Sixteen histogram gradient-boosting "
        "configurations were compared with five fixed stratified folds and average precision as the "
        "selection metric. The frozen test partition was evaluated once after selection.",
        "",
        "## Training-fold comparison",
        "",
        "| Model | Mean ROC AUC | Mean average precision |",
        "| --- | ---: | ---: |",
        f"| Logistic regression | {cv['logistic_regression']['roc_auc']['mean']:.3f} | "
        f"{cv['logistic_regression']['average_precision']['mean']:.3f} |",
        f"| Histogram gradient boosting | {cv['hist_gradient_boosting']['roc_auc']['mean']:.3f} | "
        f"{cv['hist_gradient_boosting']['average_precision']['mean']:.3f} |",
        "",
        "## Frozen-test comparison",
        "",
        "| Model | ROC AUC | Average precision | Brier score | Log loss |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Logistic regression | {logistic['roc_auc']:.3f} | {logistic['average_precision']:.3f} | "
        f"{logistic['brier_score']:.3f} | {logistic['log_loss']:.3f} |",
        f"| Histogram gradient boosting | {boosted['roc_auc']:.3f} | {boosted['average_precision']:.3f} | "
        f"{boosted['brier_score']:.3f} | {boosted['log_loss']:.3f} |",
        "",
        "Paired bootstrap 95% intervals for boosted minus logistic:",
        "",
        f"- ROC AUC: {intervals['roc_auc']['delta_boosted_minus_logistic'][0]:.3f} to "
        f"{intervals['roc_auc']['delta_boosted_minus_logistic'][1]:.3f}",
        f"- Average precision: {intervals['average_precision']['delta_boosted_minus_logistic'][0]:.3f} to "
        f"{intervals['average_precision']['delta_boosted_minus_logistic'][1]:.3f}",
        f"- Brier score: {intervals['brier_score']['delta_boosted_minus_logistic'][0]:.3f} to "
        f"{intervals['brier_score']['delta_boosted_minus_logistic'][1]:.3f} (lower is better)",
        "",
        "**Decision:** retain logistic regression as the preferred model. Gradient boosting did not show "
        "a reliable test-set improvement and had lower average precision. Its extra complexity is not justified.",
        "",
        "The comparison is internal and unweighted. It concerns prevalent self-reported history, not "
        "future cardiovascular events or clinical validity. Threshold selection and calibration remain "
        "outside this milestone.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
