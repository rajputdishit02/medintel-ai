# MedIntel AI
### Explainable Clinical Analytics & Medical Research Intelligence

A research portfolio project exploring cardiovascular health through real clinical survey data, transparent statistical analysis, and reproducible machine learning.

**Status: V0.2 complete.** The repository includes verified acquisition, a checksum manifest, reproducible adult-cohort construction, and survey-aware cardiovascular risk-factor estimates. Predictive models and applications remain planned; no model-performance results are claimed.

## Research direction

Start with CDC NHANES August 2021–August 2023: integrate demographic, examination, laboratory, and questionnaire tables rather than train on a pre-cleaned disease CSV. The initial question is: **How do cardiovascular measurements and their missingness vary across adult groups, and how are these measurements associated with reported cardiovascular disease history?**

NHANES is cross-sectional. A model of reported disease history would estimate a contemporaneous association, not prospective cardiovascular risk. The modelling target will be frozen after the cohort and leakage review. This project is for research and education, not diagnosis, treatment, or individual clinical decisions.

## What this project will demonstrate

| Area | Planned evidence |
| --- | --- |
| Data science | Documented multi-table joins, missingness analysis, survey-aware statistics, reproducible experiments |
| Medical data science | Explicit eligibility, codebook-driven definitions, survey design, bias and uncertainty reporting |
| AI engineering | Tested pipelines first; later a model API and a citation-grounded literature retrieval system |
| Explainable ML | Baselines, calibration, threshold analysis, subgroup evaluation, and SHAP with limitations |

## Quick start

Requires Python 3.11 or newer. Run from the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m medintel --version
python -m medintel sources
python -m medintel acquire --data-dir data
python -m medintel verify-data --data-dir data
python -m medintel build-cohort --data-dir data
python -m medintel eda --data-dir data
python -m unittest discover -s tests -v
```

The `sources` command is offline. `acquire` downloads nine official XPT files, validates their participant keys, and records checksums and provenance. `build-cohort` creates a local participant-level CSV plus committed aggregate metadata. Raw and processed participant data remain ignored by Git.

## Repository map

```text
src/medintel/       Small Python package and source catalogue
tests/             Offline catalogue and command-line checks
docs/              Data protocol and staged roadmap
data/              Local raw and processed data (contents ignored)
notebooks/         Conventions for future exploratory work
reports/           Conventions for future findings and figures
.github/workflows/ Foundation checks on Python 3.11 and 3.12
```

## Data and reproducibility

The source is [CDC's August 2021–August 2023 catalogue](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023). See the [data protocol](docs/data-protocol.md) for proposed components, cohort rules, provenance requirements, and analysis decisions. No participant records or fabricated findings are included.

Each future data acquisition must record source URL, retrieval time, byte count, SHA-256, and codebook revision. Retain original XPT files locally; derive versioned analysis tables through code. Population estimates will require appropriate survey weights and design-aware uncertainty. Ordinary unweighted summaries will be labelled as sample descriptions.

## Validated V0.2 result

The complete-measurement analysis contains 5,249 adults after excluding pregnant participants. It reproduces the published [NCHS Data Brief 540](https://www.cdc.gov/nchs/products/databriefs/db540.htm) benchmark: 36.4% had no defined cardiovascular risk factors, 34.9% had one, and 28.7% had two or more. Sex- and age-stratified estimates and Taylor-linearized standard errors also match the published tables.

See the [EDA report](reports/v0.2-eda.md) and [complete prevalence table](reports/tables/cvd-risk-factor-prevalence.csv). This is a reproducibility benchmark using cross-sectional population estimates, not a diagnostic or predictive result.

## Roadmap

| Milestone | Deliverable | State |
| --- | --- | --- |
| V0.1 | Repository, source catalogue, protocol, foundation checks | Complete |
| V0.2 | Verified ingestion, cohort flow, EDA and statistics | Complete |
| V0.3 | Target definition, leakage audit, baseline ML | Planned |
| V0.4 | Advanced model comparison and tuning | Planned |
| V0.5 | Calibration, thresholds, SHAP and subgroup analysis | Planned |
| V0.6 | FastAPI inference contract | Planned |
| V0.7 | Medical literature RAG with citation evaluation | Planned |
| V0.8 | Streamlit research interface | Planned |
| V0.9 | Integration tests, Docker and release checks | Planned |
| V1.0 | Documented portfolio release and deployment | Planned |

See [acceptance criteria](docs/roadmap.md). Later technologies are intentions, not current capabilities. The research assistant will be evaluated separately from the statistical models.

## Development

Keep exploratory notebooks thin and move reusable logic into `src/medintel`. Use small commits describing a complete change. Run the offline checks before committing. Never commit credentials, local environments, participant-level data, or trained model binaries. Report failures and negative findings alongside successful experiments.

## Limitations and responsible use

Self-reported history, missing measurements, survey nonresponse, and treatment after diagnosis can affect conclusions. Associations and SHAP attributions do not establish causality. Future predictive performance must be assessed on held-out data; it must not be presented as clinical validation. No clinical deployment or patient input is supported.

## Sources and reuse

- [NHANES release catalogue](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023)
- [Demographics documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm)
- [Medical conditions documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.htm)
- [NHANES analytic tutorials](https://wwwn.cdc.gov/nchs/nhanes/tutorials/default.aspx)

Sources reviewed September 29, 2026. CDC/NCHS is the data provider and does not endorse this project. No open-source licence has been selected yet; public visibility alone does not grant reuse rights. Any future code licence will be separate from the source data's terms.
