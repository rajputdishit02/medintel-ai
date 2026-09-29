"""Survey-aware descriptive analysis for the V0.2 cohort."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

RISK_SOURCE = "https://www.cdc.gov/nchs/products/databriefs/db540.htm"


def survey_ratio(
    frame: pd.DataFrame,
    outcome: pd.Series,
    domain: pd.Series | None = None,
) -> dict[str, float | int]:
    """Estimate a weighted proportion and Taylor-linearized standard error."""
    domain = pd.Series(True, index=frame.index) if domain is None else domain.fillna(False)
    valid = frame["WTPH2YR"].gt(0) & outcome.notna()
    weights = frame["WTPH2YR"].where(valid, 0.0)
    domain_weight = weights * domain.astype(float)
    denominator = float(domain_weight.sum())
    if denominator <= 0:
        raise ValueError("survey domain has no positive-weight observations")
    estimate = float((domain_weight * outcome.astype(float).fillna(0)).sum() / denominator)
    linearized = domain_weight * (outcome.astype(float).fillna(0) - estimate) / denominator
    work = frame[["SDMVSTRA", "SDMVPSU"]].copy()
    work["linearized"] = linearized
    psu_totals = work.groupby(["SDMVSTRA", "SDMVPSU"], observed=True)["linearized"].sum()
    variance = 0.0
    strata = 0
    psus = 0
    for _, totals in psu_totals.groupby(level=0):
        values = totals.to_numpy()
        count = len(values)
        if count < 2:
            raise ValueError("each variance stratum must contain at least two PSUs")
        variance += count / (count - 1) * float(((values - values.mean()) ** 2).sum())
        strata += 1
        psus += count
    standard_error = math.sqrt(variance)
    return {
        "unweighted_n": int((valid & domain).sum()),
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": max(0.0, estimate - 1.96 * standard_error),
        "ci_high": min(1.0, estimate + 1.96 * standard_error),
        "design_df": psus - strata,
    }


def prepare_risk_sample(cohort: pd.DataFrame) -> pd.DataFrame:
    required = [
        "mean_systolic_bp", "mean_diastolic_bp", "LBXTC", "LBDHDD",
        "LBXGH", "BMXBMI", "WTPH2YR", "SDMVSTRA", "SDMVPSU",
    ]
    complete = cohort[required].notna().all(axis=1)
    not_pregnant = cohort["RIDEXPRG"].ne(1) | cohort["RIDEXPRG"].isna()
    sample = cohort.loc[complete & not_pregnant].copy()
    sample["high_blood_pressure"] = (
        sample["mean_systolic_bp"].ge(130) | sample["mean_diastolic_bp"].ge(80)
    )
    sample["non_hdl_cholesterol"] = sample["LBXTC"] - sample["LBDHDD"]
    sample["high_blood_lipids"] = sample["non_hdl_cholesterol"].ge(190)
    sample["high_glucose"] = sample["LBXGH"].ge(6.5)
    sample["high_bmi"] = sample["BMXBMI"].ge(30)
    factors = ["high_blood_pressure", "high_blood_lipids", "high_glucose", "high_bmi"]
    sample["risk_factor_count"] = sample[factors].sum(axis=1)
    sample["risk_factor_group"] = sample["risk_factor_count"].map(
        lambda count: "none" if count == 0 else "one" if count == 1 else "two_or_more"
    )
    sample["sex"] = sample["RIAGENDR"].map({1.0: "men", 2.0: "women"})
    sample["age_group"] = pd.cut(
        sample["RIDAGEYR"], bins=[19, 39, 59, float("inf")],
        labels=["20-39", "40-59", "60+"],
    )
    return sample


def prevalence_table(sample: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    domains = [("all", "all", pd.Series(True, index=sample.index))]
    domains += [("sex", value, sample["sex"].eq(value)) for value in ("men", "women")]
    domains += [
        ("age", value, sample["age_group"].eq(value))
        for value in ("20-39", "40-59", "60+")
    ]
    for domain_type, domain_value, domain in domains:
        for category in ("none", "one", "two_or_more"):
            result = survey_ratio(sample, sample["risk_factor_group"].eq(category), domain)
            rows.append({
                "domain": domain_type,
                "group": domain_value,
                "risk_factors": category,
                "unweighted_n": result["unweighted_n"],
                "percent": round(100 * float(result["estimate"]), 1),
                "standard_error": round(100 * float(result["standard_error"]), 1),
                "ci_low": round(100 * float(result["ci_low"]), 1),
                "ci_high": round(100 * float(result["ci_high"]), 1),
                "design_df": result["design_df"],
            })
    return pd.DataFrame(rows)


def write_eda(data_dir: Path, reports_dir: Path) -> tuple[Path, Path]:
    cohort = pd.read_csv(data_dir / "processed" / "adult-cardiovascular-cohort.csv")
    sample = prepare_risk_sample(cohort)
    table = prevalence_table(sample)
    table_path = reports_dir / "tables" / "cvd-risk-factor-prevalence.csv"
    report_path = reports_dir / "v0.2-eda.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    overall = table.loc[table["domain"].eq("all")]
    lines = [
        "# V0.2 cardiovascular risk-factor EDA",
        "",
        f"Analysis sample: **{len(sample):,} adults** with complete measurements; pregnant participants were excluded.",
        "",
        "| Risk-factor count | Weighted percent | Standard error | 95% normal CI |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {"none": "None", "one": "One", "two_or_more": "Two or more"}
    for row in overall.itertuples():
        lines.append(
            f"| {labels[row.risk_factors]} | {row.percent:.1f}% | {row.standard_error:.1f} pp | "
            f"{row.ci_low:.1f}%-{row.ci_high:.1f}% |"
        )
    lines += [
        "",
        "Definitions follow [NCHS Data Brief 540](" + RISK_SOURCE + "): blood pressure >=130/80 mmHg, "
        "non-HDL cholesterol >=190 mg/dL, HbA1c >=6.5%, and BMI >=30 kg/m2.",
        "",
        "Estimates use the two-year phlebotomy weight and Taylor-linearized standard errors from the masked strata and PSU fields. "
        "Confidence intervals use a normal critical value. This implementation reproduces a public benchmark for validation; "
        "it is not a substitute for clinical interpretation.",
        "",
        "The complete sex- and age-stratified table is in `reports/tables/cvd-risk-factor-prevalence.csv`. "
        "Counts are unweighted; percentages describe the target population represented by the survey weights.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata_path = data_dir / "metadata" / "eda-summary.json"
    complete = cohort[[
        "mean_systolic_bp", "mean_diastolic_bp", "LBXTC", "LBDHDD",
        "LBXGH", "BMXBMI", "WTPH2YR", "SDMVSTRA", "SDMVPSU",
    ]].notna().all(axis=1)
    metadata_path.write_text(json.dumps({
        "analysis_sample_n": len(sample),
        "adult_pregnant_n": int(cohort["RIDEXPRG"].eq(1).sum()),
        "complete_pregnant_excluded_n": int((complete & cohort["RIDEXPRG"].eq(1)).sum()),
        "definition_source": RISK_SOURCE,
        "weight": "WTPH2YR",
        "variance_method": "Taylor linearization with masked strata and PSUs",
    }, indent=2) + "\n", encoding="utf-8")
    return report_path, table_path
