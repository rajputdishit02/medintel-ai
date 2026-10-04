"""Leakage-conscious baseline models for prevalent reported CVD history."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 417
NUMERIC_FEATURES = [
    "RIDAGEYR", "INDFMPIR", "mean_systolic_bp", "mean_diastolic_bp",
    "BMXBMI", "BMXWAIST", "LBXTC", "LBDHDD", "LBXGH",
]
CATEGORICAL_FEATURES = ["RIAGENDR", "RIDRETH3", "ever_smoked"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_model_frame(cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    labelled = cohort.loc[
        cohort["cvd_history"].isin(["positive", "negative"])
    ].copy().reset_index(drop=True)
    labelled["ever_smoked"] = labelled["SMQ020"].map({1.0: "yes", 2.0: "no"})
    for column in ("RIAGENDR", "RIDRETH3"):
        labelled[column] = labelled[column].astype("Int64").astype("string")
    target = labelled["cvd_history"].eq("positive").astype(int)
    return labelled[FEATURES], target, labelled["SEQN"]


def preprocessing(*, dense: bool = False) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore", drop="if_binary", sparse_output=not dense,
        )),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def evaluate(y_true: pd.Series, probability) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability)),
        "sensitivity_at_0_5": float(tp / (tp + fn)),
        "specificity_at_0_5": float(tn / (tn + fp)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def run_baselines(data_dir: Path, reports_dir: Path, models_dir: Path) -> Path:
    cohort_path = data_dir / "processed" / "adult-cardiovascular-cohort.csv"
    cohort = pd.read_csv(cohort_path)
    features, target, identifiers = prepare_model_frame(cohort)
    indices = pd.Series(range(len(features)), index=features.index)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=target,
    )
    x_train, x_test = features.loc[train_idx], features.loc[test_idx]
    y_train, y_test = target.loc[train_idx], target.loc[test_idx]

    models = {
        "dummy_prior": Pipeline([
            ("preprocess", preprocessing()),
            ("model", DummyClassifier(strategy="prior")),
        ]),
        "logistic_regression": Pipeline([
            ("preprocess", preprocessing()),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
    }
    results: dict[str, object] = {}
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        results[name] = evaluate(y_test, probability)
        joblib.dump(model, models_dir / f"{name}.joblib")

    split_records = pd.DataFrame({
        "SEQN": identifiers,
        "partition": "train",
    })
    split_records.loc[test_idx, "partition"] = "test"
    split_hash = hashlib.sha256(
        split_records.sort_values("SEQN").to_csv(index=False).encode("utf-8")
    ).hexdigest()
    split_path = data_dir / "processed" / "v0.3-model-split.csv"
    split_records.sort_values("SEQN").to_csv(split_path, index=False)
    metadata = {
        "target": "cvd_history_positive",
        "interpretation": "prevalent self-reported history; not future risk",
        "seed": SEED,
        "data_sha256": file_sha256(cohort_path),
        "split_sha256": split_hash,
        "local_split_path": str(split_path),
        "sample_n": len(target),
        "positive_n": int(target.sum()),
        "negative_n": int((1 - target).sum()),
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "features": FEATURES,
        "excluded_for_unknown_target": int(cohort["cvd_history"].eq("unknown").sum()),
        "evaluation_weighting": "unweighted",
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "results": results,
    }
    output = reports_dir / "metrics" / "v0.3-baselines.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    dummy = results["dummy_prior"]
    logistic = results["logistic_regression"]
    report = reports_dir / "v0.3-baseline.md"
    report.write_text(
        "# V0.3 baseline modelling report\n\n"
        f"The frozen dataset contains **{len(target):,} labelled adults**: "
        f"{int(target.sum()):,} positive and {int((1 - target).sum()):,} negative outcomes. "
        f"A fixed stratified split assigned {len(train_idx):,} records to training and "
        f"{len(test_idx):,} to the untouched test partition.\n\n"
        "| Model | ROC AUC | Average precision | Brier score | Log loss | Sensitivity at 0.5 | Specificity at 0.5 |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| Class-prior dummy | {dummy['roc_auc']:.3f} | {dummy['average_precision']:.3f} | "
        f"{dummy['brier_score']:.3f} | {dummy['log_loss']:.3f} | "
        f"{dummy['sensitivity_at_0_5']:.3f} | {dummy['specificity_at_0_5']:.3f} |\n"
        f"| Logistic regression | {logistic['roc_auc']:.3f} | {logistic['average_precision']:.3f} | "
        f"{logistic['brier_score']:.3f} | {logistic['log_loss']:.3f} | "
        f"{logistic['sensitivity_at_0_5']:.3f} | {logistic['specificity_at_0_5']:.3f} |\n\n"
        "The 0.5 threshold is an unevaluated reference point. Its low sensitivity makes it unsuitable "
        "for a clinical interpretation. No operating threshold has been selected.\n\n"
        "These are unweighted internal test-set results for a cross-sectional association model of "
        "self-reported history. They do not establish future-risk prediction, clinical validity, or causality. "
        "See `docs/modeling-protocol.md` for the feature and leakage decisions.\n",
        encoding="utf-8",
    )
    return output
