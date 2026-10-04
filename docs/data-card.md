# Data card

## Source and scope

MedIntel AI uses the public CDC National Health and Nutrition Examination Survey August 2021–August 2023 cycle. The pipeline joins demographic, examination, laboratory, and questionnaire components by participant identifier. Source URLs, retrieval timestamps, byte counts, and SHA-256 hashes are recorded locally.

## Cohort and outcome

The descriptive benchmark includes 5,249 non-pregnant adults aged 20 years or older with complete measurements required by the published cardiovascular risk-factor definition. The modelling outcome is prevalent self-reported history of coronary heart disease, angina, heart attack, or stroke. Unknown responses remain unknown and are excluded from supervised modelling.

## Sensitive attributes

Age, sex, and NHANES race and ethnicity categories are used for adjustment and subgroup auditing. These categories are survey variables with social and historical context; they are not biological explanations of model behaviour.

## Appropriate use

The data supports education, epidemiological analysis, reproducibility exercises, and methodological research. NHANES survey design variables and weights are required for population estimates.

## Inappropriate use

Do not use this dataset or project for diagnosis, treatment, screening, insurance, employment, or decisions about an individual. The cross-sectional outcome cannot establish future cardiovascular risk or causal effects.

## Distribution

Participant-level files, derived participant tables, split assignments, and trained models are excluded from Git. Aggregate tables and metrics are committed when they do not identify participants. CDC data terms remain separate from this repository's MIT-licensed source code.
