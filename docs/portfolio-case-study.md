# MedIntel AI portfolio case study

## One-line description

Built an end-to-end clinical analytics and medical research intelligence platform using CDC NHANES data, explainable machine learning, a validated API, PubMed retrieval, and a deployed Streamlit interface.

## Problem

Many healthcare machine-learning demonstrations begin with a small, already-cleaned CSV and report a single accuracy score. MedIntel AI was designed to demonstrate the harder parts of medical data science: integrating documented clinical tables, defining a defensible cohort and outcome, respecting survey design, testing calibration and subgroup behaviour, and communicating limitations.

## Approach

The project joins nine NHANES 2021–2023 components across demographics, examinations, laboratory measurements, and questionnaires. It records source provenance and checksums, preserves unknown outcome responses, and separates participant-level artifacts from committed aggregate results.

The modelling workflow fixes the target and split before comparison, establishes a class-prior baseline, and compares logistic regression with tuned histogram gradient boosting. Because the more complex model did not improve frozen-test performance, the interpretable logistic model was retained. Calibration, threshold scenarios, SHAP attributions, and subgroup results are reported separately.

A FastAPI contract validates clinical ranges and labels every response as research-only. The Streamlit interface presents the published evidence and a PubMed assistant. Literature answers use local retrieval, numbered sources, citation validation, and abstention; OpenAI synthesis is optional.

## Evidence

- Reproduced a published CDC cardiovascular risk-factor benchmark in 5,249 adults.
- Achieved test ROC AUC 0.793 and average precision 0.390 for prevalent self-reported cardiovascular disease history.
- Found no meaningful benefit from gradient boosting over logistic regression.
- Identified weak performance in adults aged 20–39, including only 12 positive test cases.
- Added 20 automated tests across survey estimation, cohort rules, modelling, API validation, retrieval, citations, and Streamlit rendering.
- Passed CI on Python 3.11 and 3.12 plus a clean Docker build.

## Technical decisions worth discussing

1. **Outcome framing:** The target is cross-sectional self-reported history, so outputs are never described as future-event predictions.
2. **Model selection:** Complexity was rejected when it did not add reliable held-out value.
3. **Thresholds:** The 80% sensitivity scenario is explicitly exploratory and is not a clinical recommendation.
4. **Subgroup evidence:** Poor young-adult results are reported rather than hidden behind aggregate performance.
5. **Data governance:** Participant rows, split assignments, credentials, and the trained model stay out of the public repository and deployment.
6. **RAG safeguards:** Retrieved abstracts are treated as evidence rather than instructions; answers require valid source numbers or fall back to extracts.

## Interview summary

“I wanted this project to show progression beyond standard disease-classification notebooks. I started from official NHANES source tables, reproduced a published CDC benchmark, froze a defensible modelling design, and found that gradient boosting did not beat logistic regression. I retained the simpler model, evaluated calibration and subgroup behaviour, and exposed the results through a guarded API and Streamlit app. I also added a PubMed retrieval assistant that cites its evidence and abstains when retrieval is weak. The most important result was not the AUC; it was demonstrating reproducible decisions and communicating where the model should not be used.”

## Suggested résumé bullets

- Built and deployed an end-to-end clinical analytics platform integrating nine CDC NHANES tables, survey-aware estimation, explainable ML, FastAPI, Streamlit, and PubMed retrieval.
- Reproduced a published CDC cardiovascular benchmark in 5,249 adults and developed a calibrated logistic model with 0.793 test ROC AUC, retaining it after tuned gradient boosting showed no reliable improvement.
- Implemented SHAP interpretation, threshold and subgroup audits, strict API validation, citation-grounded research retrieval, 20 automated tests, multi-version CI, and Docker packaging.

## Links

- [Live application](https://medintel-cvd-research.streamlit.app/)
- [Source repository](https://github.com/rajputdishit02/medintel-ai)
- [Release](https://github.com/rajputdishit02/medintel-ai/releases/tag/v1.0.2)
