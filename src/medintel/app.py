"""Native Streamlit research interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from medintel.api import ClinicalFeatures, ModelRuntime
from medintel.literature import acquire_pubmed, answer_question

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "reports" / "metrics" / "v0.5-interpretation.json"
CORPUS = ROOT / "data" / "literature" / "pubmed-corpus.jsonl"


@st.cache_data
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource
def load_runtime() -> ModelRuntime:
    return ModelRuntime()


def overview() -> None:
    st.title("MedIntel AI")
    st.caption("Explainable clinical analytics and medical research intelligence")
    st.info(
        "Research and education only. Results describe associations in a cross-sectional "
        "NHANES cohort and must not guide diagnosis or treatment.",
        icon=":material/science:",
    )
    metrics = load_json(str(METRICS))["frozen_test"]
    columns = st.columns(4)
    columns[0].metric("Adults analysed", "5,249")
    columns[1].metric("Test ROC AUC", f"{metrics['roc_auc']:.3f}")
    columns[2].metric("Average precision", f"{metrics['average_precision']:.3f}")
    columns[3].metric("Test sensitivity", f"{metrics['sensitivity']:.1%}")
    st.subheader("What the project demonstrates")
    st.markdown(
        """
        - Joins and validates nine official NHANES demographic, examination, laboratory, and questionnaire tables.
        - Reproduces a published CDC cardiovascular risk-factor benchmark with survey-aware uncertainty.
        - Compares interpretable and boosted models, then retains logistic regression when complexity adds no reliable benefit.
        - Audits calibration, thresholds, SHAP attributions, and subgroup performance before exposing a guarded API.
        - Retrieves PubMed evidence and requires traceable citations or an explicit abstention.
        """
    )
    figure = ROOT / "reports" / "figures" / "v0.5-calibration.png"
    if figure.exists():
        st.image(str(figure), caption="Calibration on the frozen test set", width="stretch")
    st.warning(
        "The model estimates association with previously reported cardiovascular disease, "
        "not the probability of a future event.",
        icon=":material/warning:",
    )


def risk_explorer() -> None:
    st.title("Research risk explorer")
    st.write(
        "Explore how the retained model responds to a hypothetical feature profile. "
        "This output is not a clinical risk score."
    )
    runtime = load_runtime()
    if not runtime.available:
        st.error(
            "The local trained model is unavailable. Run the modelling and interpretation steps first.",
            icon=":material/model_training:",
        )
        return
    with st.form("risk-profile"):
        left, right = st.columns(2)
        age = left.slider("Age", 20, 80, 50)
        sex = left.selectbox("Sex recorded by NHANES", ["female", "male"])
        race = left.selectbox("Race and ethnicity category", list({
            "non_hispanic_white": 1, "non_hispanic_black": 1,
            "non_hispanic_asian": 1, "mexican_american": 1,
            "other_hispanic": 1, "other_or_multiracial": 1,
        }))
        systolic = left.number_input("Systolic blood pressure (mmHg)", 70.0, 250.0, 125.0)
        diastolic = left.number_input("Diastolic blood pressure (mmHg)", 30.0, 150.0, 78.0)
        bmi = right.number_input("BMI (kg/m²)", 10.0, 80.0, 27.0)
        waist = right.number_input("Waist circumference (cm)", 40.0, 200.0, 90.0)
        total = right.number_input("Total cholesterol (mg/dL)", 50.0, 500.0, 190.0)
        hdl = right.number_input("HDL cholesterol (mg/dL)", 5.0, 200.0, 50.0)
        hba1c = right.number_input("HbA1c (%)", 2.0, 20.0, 5.5)
        smoked = right.selectbox("Ever smoked 100 cigarettes", ["Unknown", "No", "Yes"])
        submitted = st.form_submit_button("Run research estimate", icon=":material/analytics:")
    if submitted:
        try:
            features = ClinicalFeatures(
                age_years=age, sex=sex, race_ethnicity=race,
                systolic_bp=systolic, diastolic_bp=diastolic, bmi=bmi,
                waist_cm=waist, total_cholesterol_mg_dl=total,
                hdl_cholesterol_mg_dl=hdl, hba1c_percent=hba1c,
                ever_smoked=None if smoked == "Unknown" else smoked == "Yes",
            )
            probability = float(runtime.model.predict_proba(features.as_model_frame())[0, 1])
            st.metric("Model estimate", f"{probability:.1%}")
            st.caption(
                "Estimated association with prevalent self-reported history in this research cohort. "
                "The exploratory threshold is not a treatment or screening boundary."
            )
        except ValueError as error:
            st.error(str(error))


def research_assistant() -> None:
    st.title("Medical research assistant")
    st.write("Ask about cardiovascular modelling, calibration, explainability, or risk factors.")
    question = st.chat_input("Ask a research question")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving PubMed evidence…"):
                if not CORPUS.exists():
                    acquire_pubmed(
                        CORPUS,
                        ROOT / "data" / "metadata" / "literature-summary.json",
                    )
                result = answer_question(question, CORPUS)
            st.markdown(result["answer"])
            st.caption(f"Answer mode: {result['mode'].replace('_', ' ')}")
            if result["sources"]:
                with st.expander("PubMed sources", expanded=True):
                    for source in result["sources"]:
                        st.markdown(
                            f"[{source['citation']}] [{source['title']}]({source['url']}) "
                            f"({source['year']})"
                        )


def methods() -> None:
    st.title("Methods and limitations")
    st.subheader("Study design")
    st.write(
        "CDC NHANES August 2021–August 2023 is a cross-sectional, nationally representative "
        "health survey. The analysis uses documented eligibility, survey weights, masked strata, "
        "and primary sampling units."
    )
    st.subheader("Model evidence")
    comparison = load_json(str(ROOT / "reports" / "metrics" / "v0.4-model-comparison.json"))
    rows = []
    for name, values in comparison["frozen_test"].items():
        rows.append({"Model": name.replace("_", " ").title(), "ROC AUC": values["roc_auc"], "Average precision": values["average_precision"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Known limitations")
    st.markdown(
        """
        - The outcome is self-reported prior disease and does not measure incident future events.
        - Missingness, survey nonresponse, and treatment after diagnosis can affect associations.
        - Internal test performance is not external or clinical validation.
        - The youngest subgroup had few positive cases and materially weaker performance.
        - SHAP values describe model behaviour; they do not establish causal effects.
        """
    )


def render() -> None:
    st.set_page_config(page_title="MedIntel AI", page_icon=":material/cardiology:", layout="wide")
    page = st.navigation({
        "Explore": [
            st.Page(overview, title="Project overview", icon=":material/home:", default=True),
            st.Page(risk_explorer, title="Risk explorer", icon=":material/monitor_heart:"),
            st.Page(research_assistant, title="Research assistant", icon=":material/library_books:"),
        ],
        "Documentation": [st.Page(methods, title="Methods and limitations", icon=":material/description:")],
    })
    page.run()


def run() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "streamlit_app.py")], check=True)
