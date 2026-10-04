"""Calibration, threshold, subgroup, and SHAP analysis for V0.5."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_matplotlib_cache = Path(tempfile.gettempdir()) / "medintel-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from medintel.comparison import frozen_split
from medintel.modeling import SEED, preprocessing

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", preprocessing()),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def select_high_sensitivity_threshold(
    y_true: pd.Series, probability: np.ndarray, minimum_sensitivity: float = 0.80,
) -> dict[str, float]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probability)
    valid = np.flatnonzero(true_positive_rate >= minimum_sensitivity)
    if not len(valid):
        raise ValueError("no threshold meets the sensitivity target")
    specificity = 1 - false_positive_rate
    best_specificity = specificity[valid].max()
    candidates = valid[specificity[valid] == best_specificity]
    selected = candidates[np.argmax(thresholds[candidates])]
    return {
        "threshold": float(thresholds[selected]),
        "training_sensitivity": float(true_positive_rate[selected]),
        "training_specificity": float(specificity[selected]),
        "minimum_sensitivity": minimum_sensitivity,
    }


def threshold_metrics(
    y_true: pd.Series, probability: np.ndarray, threshold: float,
) -> dict[str, float | int]:
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "unweighted_n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else 0.0,
        "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else 0.0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def subgroup_table(
    x_test: pd.DataFrame, y_test: pd.Series, probability: np.ndarray, threshold: float,
) -> pd.DataFrame:
    groups: list[tuple[str, str, pd.Series]] = [
        ("sex", "men", x_test["RIAGENDR"].eq("1")),
        ("sex", "women", x_test["RIAGENDR"].eq("2")),
        ("age", "20-39", x_test["RIDAGEYR"].between(20, 39)),
        ("age", "40-59", x_test["RIDAGEYR"].between(40, 59)),
        ("age", "60+", x_test["RIDAGEYR"].ge(60)),
    ]
    rows = []
    for dimension, group, mask in groups:
        y_group = y_test.loc[mask]
        p_group = probability[np.asarray(mask)]
        metrics = threshold_metrics(y_group, p_group, threshold)
        rows.append({
            "dimension": dimension,
            "group": group,
            **metrics,
            "roc_auc": float(roc_auc_score(y_group, p_group)),
            "average_precision": float(average_precision_score(y_group, p_group)),
            "brier_score": float(brier_score_loss(y_group, p_group)),
        })
    return pd.DataFrame(rows)


def shap_importance(
    model: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame,
) -> pd.DataFrame:
    transform = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    rng = np.random.default_rng(SEED)
    background_rows = rng.choice(len(x_train), size=min(500, len(x_train)), replace=False)
    transformed_train = transform.transform(x_train.iloc[background_rows])
    transformed_test = transform.transform(x_test)
    if hasattr(transformed_train, "toarray"):
        transformed_train = transformed_train.toarray()
        transformed_test = transformed_test.toarray()
    masker = shap.maskers.Independent(
        transformed_train, max_samples=len(transformed_train),
    )
    explainer = shap.LinearExplainer(estimator, masker)
    explanation = explainer(transformed_test)
    names = transform.get_feature_names_out()
    result = pd.DataFrame({
        "feature": names,
        "mean_absolute_shap_log_odds": np.abs(explanation.values).mean(axis=0),
        "coefficient": estimator.coef_[0],
    }).sort_values("mean_absolute_shap_log_odds", ascending=False, ignore_index=True)
    result["display_feature"] = result["feature"].map(display_feature_name)
    return result


def display_feature_name(name: str) -> str:
    cleaned = name.replace("numeric__", "").replace("categorical__", "")
    cleaned = cleaned.replace("missingindicator_", "Missing: ")
    labels = {
        "RIDAGEYR": "Age",
        "LBXTC": "Total cholesterol",
        "INDFMPIR": "Income-to-poverty ratio",
        "BMXWAIST": "Waist circumference",
        "BMXBMI": "BMI",
        "LBXGH": "HbA1c",
        "RIAGENDR_2": "Sex: women",
        "ever_smoked_yes": "Ever smoked",
        "mean_systolic_bp": "Mean systolic BP",
        "mean_diastolic_bp": "Mean diastolic BP",
        "LBDHDD": "HDL cholesterol",
    }
    if cleaned.startswith("Missing: "):
        variable = cleaned.removeprefix("Missing: ")
        return "Missing: " + labels.get(variable, variable)
    return labels.get(cleaned, cleaned)


def run_interpretation(data_dir: Path, reports_dir: Path, models_dir: Path) -> Path:
    x_train, x_test, y_train, y_test = frozen_split(data_dir)
    outer_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        "uncalibrated": logistic_pipeline(),
        "sigmoid": CalibratedClassifierCV(logistic_pipeline(), method="sigmoid", cv=3),
        "isotonic": CalibratedClassifierCV(logistic_pipeline(), method="isotonic", cv=3),
    }
    oof_probabilities: dict[str, np.ndarray] = {}
    selection: dict[str, dict[str, float]] = {}
    for name, candidate in candidates.items():
        probability = cross_val_predict(
            candidate, x_train, y_train, cv=outer_folds,
            method="predict_proba", n_jobs=-1,
        )[:, 1]
        oof_probabilities[name] = probability
        selection[name] = {
            "brier_score": float(brier_score_loss(y_train, probability)),
            "log_loss": float(log_loss(y_train, probability)),
        }
    selected_name = min(selection, key=lambda name: selection[name]["brier_score"])
    selected_model = candidates[selected_name]
    threshold = select_high_sensitivity_threshold(
        y_train, oof_probabilities[selected_name], minimum_sensitivity=0.80,
    )

    selected_model.fit(x_train, y_train)
    test_probability = selected_model.predict_proba(x_test)[:, 1]
    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probability)),
        "average_precision": float(average_precision_score(y_test, test_probability)),
        "brier_score": float(brier_score_loss(y_test, test_probability)),
        "log_loss": float(log_loss(y_test, test_probability)),
        **threshold_metrics(y_test, test_probability, threshold["threshold"]),
    }
    subgroup = subgroup_table(x_test, y_test, test_probability, threshold["threshold"])
    table_dir = reports_dir / "tables"
    figure_dir = reports_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    subgroup.to_csv(table_dir / "v0.5-subgroup-performance.csv", index=False)

    base_model = logistic_pipeline().fit(x_train, y_train)
    importance = shap_importance(base_model, x_train, x_test)
    importance.to_csv(table_dir / "v0.5-shap-importance.csv", index=False)
    _plot_shap(importance.head(12), figure_dir / "v0.5-shap-importance.png")
    _plot_calibration(
        y_test, x_test, selected_name, base_model, selected_model,
        figure_dir / "v0.5-calibration.png",
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, models_dir / "calibrated_logistic.joblib")

    results = {
        "calibration_selection": {
            "method": "lowest five-fold out-of-fold training Brier score",
            "candidates": selection,
            "selected": selected_name,
        },
        "threshold_selection": threshold,
        "frozen_test": test_metrics,
        "subgroup_table": "reports/tables/v0.5-subgroup-performance.csv",
        "shap_table": "reports/tables/v0.5-shap-importance.csv",
        "shap_scale": "log odds of the uncalibrated logistic model",
    }
    output = reports_dir / "metrics" / "v0.5-interpretation.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _write_report(reports_dir / "v0.5-interpretation.md", results, importance, subgroup)
    return output


def _plot_calibration(y_test, x_test, selected_name, base_model, selected_model, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#667085", label="Ideal")
    for label, model, color in [
        ("uncalibrated", base_model, "#B54708"),
        (selected_name, selected_model, "#175CD3"),
    ]:
        probability = model.predict_proba(x_test)[:, 1]
        observed, predicted = calibration_curve(
            y_test, probability, n_bins=10, strategy="quantile",
        )
        ax.plot(predicted, observed, marker="o", color=color, label=label)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed proportion", title="Frozen-test calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_shap(importance: pd.DataFrame, path: Path) -> None:
    ordered = importance.sort_values("mean_absolute_shap_log_odds")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(ordered["display_feature"], ordered["mean_absolute_shap_log_odds"], color="#175CD3")
    ax.set(xlabel="Mean |SHAP value| (log-odds scale)", title="Global logistic-model importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_report(path: Path, results, importance, subgroup) -> None:
    calibration = results["calibration_selection"]
    threshold = results["threshold_selection"]
    test = results["frozen_test"]
    lines = [
        "# V0.5 calibration, thresholds, subgroups, and SHAP",
        "",
        f"Training-only selection chose **{calibration['selected']}** calibration by lowest out-of-fold "
        f"Brier score. The exploratory threshold was {threshold['threshold']:.3f}, selected in training "
        "to reach at least 80% sensitivity while maximizing specificity.",
        "",
        "## Frozen-test performance",
        "",
        "| ROC AUC | Average precision | Brier | Sensitivity | Specificity | PPV | NPV |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {test['roc_auc']:.3f} | {test['average_precision']:.3f} | {test['brier_score']:.3f} | "
        f"{test['sensitivity']:.3f} | {test['specificity']:.3f} | "
        f"{test['positive_predictive_value']:.3f} | {test['negative_predictive_value']:.3f} |",
        "",
        "This threshold is an analytical scenario, not a clinical recommendation. Its predictive values "
        "depend on outcome prevalence in this sample.",
        "",
        "## Global explanation",
        "",
        "Top SHAP features for the underlying logistic model:",
        "",
        "| Feature | Mean absolute SHAP value |",
        "| --- | ---: |",
    ]
    for row in importance.head(8).itertuples():
        lines.append(f"| {row.display_feature} | {row.mean_absolute_shap_log_odds:.3f} |")
    lines += [
        "",
        "SHAP values describe how this fitted model distributes log-odds contributions relative to its "
        "training background. They do not measure causal effects, biological importance, or the benefit "
        "of changing a feature. Correlated clinical measurements can share or shift attribution.",
        "",
        "## Subgroup audit",
        "",
        "The sex and age audit reports unweighted internal test performance. Small positive counts make "
        "some estimates unstable; it is a diagnostic check rather than evidence of fairness or transportability.",
        "The 20-39 group contained only 12 positive cases, had ROC AUC below 0.5, and no positives "
        "crossed the selected threshold. This is a material failure mode for this sample.",
        "",
        "| Group | n | Positive n | ROC AUC | Sensitivity | Specificity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in subgroup.itertuples():
        lines.append(
            f"| {row.group} | {row.unweighted_n} | {row.positive_n} | {row.roc_auc:.3f} | "
            f"{row.sensitivity:.3f} | {row.specificity:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
