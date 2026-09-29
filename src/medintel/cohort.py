"""Construct the initial adult cardiovascular research cohort."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

COMPONENT_COLUMNS = {
    "BPXO_L": ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"],
    "BMX_L": ["BMXBMI", "BMXWAIST"],
    "TCHOL_L": ["LBXTC"],
    "HDL_L": ["LBDHDD"],
    "GHB_L": ["LBXGH"],
    "BPQ_L": ["BPQ020", "BPQ150", "BPQ080", "BPQ101D"],
    "MCQ_L": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
    "SMQ_L": ["SMQ020", "SMQ040"],
}

OUTPUT_VARIABLES = {
    "SEQN": ("identifier", "NHANES respondent sequence number"),
    "RIDAGEYR": ("years", "Age at screening"),
    "RIAGENDR": ("code", "Reported sex code; retain CDC coding"),
    "RIDRETH3": ("code", "Race and Hispanic origin code; retain CDC coding"),
    "WTMEC2YR": ("weight", "Two-year MEC examination sample weight"),
    "SDMVSTRA": ("code", "Masked variance pseudo-stratum"),
    "SDMVPSU": ("code", "Masked variance pseudo-PSU"),
    "INDFMPIR": ("ratio", "Family income-to-poverty ratio"),
    "mean_systolic_bp": ("mmHg", "Mean of available BPXOSY1-3 readings"),
    "mean_diastolic_bp": ("mmHg", "Mean of available BPXODI1-3 readings"),
    "BMXBMI": ("kg/m2", "Body mass index"),
    "BMXWAIST": ("cm", "Waist circumference"),
    "LBXTC": ("mg/dL", "Total cholesterol"),
    "LBDHDD": ("mg/dL", "HDL cholesterol"),
    "LBXGH": ("percent", "Glycohemoglobin HbA1c"),
    "BPQ020": ("code", "Ever told had high blood pressure"),
    "BPQ150": ("code", "Currently taking prescribed hypertension medication"),
    "BPQ080": ("code", "Ever told blood cholesterol was high"),
    "BPQ101D": ("code", "Currently taking prescribed cholesterol medication"),
    "SMQ020": ("code", "Smoked at least 100 cigarettes in life"),
    "SMQ040": ("code", "Current smoking frequency"),
    "cvd_history": ("category", "Reported CVD history: positive, negative, or unknown"),
}


def _read_component(raw_dir: Path, code: str, columns: list[str]) -> pd.DataFrame:
    path = raw_dir / f"{code}.xpt"
    frame = pd.read_sas(path, format="xport")
    required = ["SEQN", *columns]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{code}: missing columns {missing}")
    if frame["SEQN"].isna().any() or frame["SEQN"].duplicated().any():
        raise ValueError(f"{code}: SEQN must be complete and unique")
    return frame[required]


def derive_cvd_history(frame: pd.DataFrame) -> pd.Series:
    """Combine five self-reported history items without treating unknown as no."""
    items = frame[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]]
    positive = items.eq(1).any(axis=1)
    negative = items.eq(2).all(axis=1)
    outcome = pd.Series("unknown", index=frame.index, dtype="string")
    outcome.loc[negative] = "negative"
    outcome.loc[positive] = "positive"
    return outcome


def build_cohort(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    demographics = pd.read_sas(raw_dir / "DEMO_L.xpt", format="xport")
    demographic_columns = [
        "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "WTMEC2YR",
        "SDMVSTRA", "SDMVPSU", "INDFMPIR",
    ]
    missing = sorted(set(demographic_columns) - set(demographics.columns))
    if missing:
        raise ValueError(f"DEMO_L: missing columns {missing}")
    if demographics["SEQN"].isna().any() or demographics["SEQN"].duplicated().any():
        raise ValueError("DEMO_L: SEQN must be complete and unique")
    cohort = demographics.loc[
        demographics["RIDAGEYR"] >= 20, demographic_columns
    ].copy()
    audit: dict[str, object] = {
        "demographics_rows": len(demographics),
        "adult_rows": len(cohort),
        "joins": {},
    }
    for code, columns in COMPONENT_COLUMNS.items():
        component = _read_component(raw_dir, code, columns)
        matched = int(cohort["SEQN"].isin(component["SEQN"]).sum())
        cohort = cohort.merge(component, on="SEQN", how="left", validate="one_to_one")
        audit["joins"][code] = {
            "source_rows": len(component),
            "adult_rows_matched": matched,
            "adult_rows_unmatched": len(cohort) - matched,
        }
    cohort["mean_systolic_bp"] = cohort[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    cohort["mean_diastolic_bp"] = cohort[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    cohort["cvd_history"] = derive_cvd_history(cohort)
    reading_columns = ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"]
    cohort = cohort.drop(columns=[*reading_columns, "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"])
    cohort = cohort[list(OUTPUT_VARIABLES)]
    audit["outcome_counts"] = {
        str(key): int(value) for key, value in cohort["cvd_history"].value_counts().items()
    }
    audit["missing_counts"] = {
        column: int(count) for column, count in cohort.isna().sum().items()
    }
    return cohort, audit


def write_cohort(data_dir: Path) -> tuple[Path, Path]:
    cohort, audit = build_cohort(data_dir / "raw" / "nhanes-2021-2023")
    processed_path = data_dir / "processed" / "adult-cardiovascular-cohort.csv"
    summary_path = data_dir / "metadata" / "cohort-summary.json"
    dictionary_path = data_dir / "metadata" / "variable-dictionary.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(processed_path, index=False)
    summary_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {"variable": variable, "unit_or_type": details[0], "description": details[1]}
            for variable, details in OUTPUT_VARIABLES.items()
        ]
    ).to_csv(dictionary_path, index=False)
    return processed_path, summary_path
