# Delivery roadmap

Build one reviewable milestone at a time. No empty API, UI, RAG, or model implementations are included in V0.1.

1. **V0.1 — foundation:** installable package, official source catalogue, protocol, README, offline checks, and meaningful Git history.
2. **V0.2 — data and statistics (complete):** acquisition/provenance, XPT validation, adult cohort, variable dictionary, join audit, missingness counts, and survey-aware descriptive analysis. The implementation reproduces the published NCHS overall, sex, and age estimates and standard errors.
3. **V0.3 — baseline (complete):** frozen target and leakage decisions, class-prior and logistic baselines, training-only preprocessing, untouched test evaluation, and recorded seeds, versions, data hash, and split hash.
4. **V0.4 — comparisons (complete):** tuned histogram gradient boosting in fixed training folds, compared it fairly with logistic regression, evaluated the frozen test partition once, and quantified paired uncertainty. Logistic regression was retained because boosting showed no reliable improvement.
5. **V0.5 — interpretation (complete):** selected sigmoid calibration and an exploratory high-sensitivity threshold with training-only predictions, evaluated the frozen test partition, audited sex and age subgroups, and produced model-dependent SHAP explanations with non-causal limits.
6. **V0.6 — API:** expose a validated, versioned research inference contract through FastAPI with contract tests and documented model limitations.
7. **V0.7 — research intelligence:** build a separate literature retrieval workflow with source metadata, grounded citations, abstention, retrieval evaluation, and answer-quality checks. Decide corpus, access terms, provider, and budget before integration.
8. **V0.8 — interface:** add a Streamlit research interface for validated analyses, model limitations, and cited literature answers.
9. **V0.9 — operations:** integration tests, Docker, reproducible environments, and deployment checks. Unit tests grow with each earlier milestone.
10. **V1.0 — release:** publish reproducible findings, a model card, a data card, setup instructions, and a demonstrable portfolio walkthrough.

Next work: design a versioned FastAPI research contract around the retained model, with strict input validation, model metadata, and contract tests for V0.6.
