# NHANES data protocol — draft for ingestion

## Scope and source of truth

Use only the August 2021–August 2023 release for the first cohort. The [official catalogue](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023) and each component's codebook govern variable definitions, eligibility, units, and missing-value codes. `python -m medintel sources` lists the proposed components. The plan is not evidence that files or variables have passed validation.

| Component | Intended use |
| --- | --- |
| DEMO_L | Cohort anchor, age, demographics, weights and survey design |
| BPXO_L | Repeated measured blood pressure |
| BMX_L | Body measurements |
| TCHOL_L / HDL_L | Lipid measurements |
| GHB_L | Glycohemoglobin |
| BPQ_L | Reported hypertension/cholesterol history and treatment context |
| MCQ_L | Reported cardiovascular disease history |
| SMQ_L | Smoking context |

Fasting glucose and LDL/triglycerides are deferred until their eligibility and subsample weighting requirements are reviewed. Do not silently restrict the main cohort to a fasting subsample.

## Proposed cohort and integration

Start with adults aged 20 or older, matching the eligibility of the candidate cardiovascular history items in [MCQ_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.htm). Confirm all eligibility rules before implementation. Anchor on DEMO_L and left-join selected participant-level components on `SEQN`. Reject null or duplicate keys before any one-to-one join. Report unmatched rows, column collisions, and row counts at every step; do not let an inner join silently define the cohort.

Preserve examination status and distinguish eligibility, nonparticipation, refusal, unknown responses, and unavailable measurements. Never globally replace every 7 or 9 with missing: use variable-specific codebooks. Keep raw values and document derived values, units, valid ranges, repeated-measure handling, and exclusions in a variable dictionary.

## Target decision before modelling

Candidate outcome: self-reported history across MCQ160b/c/d/e/f (heart failure, coronary heart disease, angina, heart attack, stroke). Review whether this heterogeneous composite is scientifically useful before adopting it. If adopted, use any valid yes as positive; require all five valid no answers for negative; otherwise retain unknown. Report item-level and composite counts. Never label unknown as negative.

This is prevalent reported history, not adjudicated disease or future events. Define predictor timing and intended interpretation. Exclude target items and derivatives from predictors. Review treatment variables and post-diagnosis measurements for leakage and reverse causation; document sensitivity analyses. Freeze the outcome, exclusions, and predictor set before comparing models.

## Survey-aware analysis

Retain the release's strata, PSU, and weight fields from [DEMO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm). Record exact field names after codebook review. Select interview, examination, or subsample weights according to the components used. Weighted point estimates alone do not provide design-correct confidence intervals. Follow [CDC tutorials](https://wwwn.cdc.gov/nchs/nhanes/tutorials/default.aspx) for domain analysis and variance estimation.

Keep population inference distinct from predictive evaluation. Explicitly document whether model fitting and evaluation are weighted and what population each metric describes. Assess subgroup sample sizes before presenting comparisons.

## Acquisition acceptance checklist

- Record official download URL, UTC retrieval time, byte count, SHA-256, and documentation revision for each file.
- Validate transport content and successful XPT parsing; fail on HTML error pages, missing columns, or invalid identifiers.
- Produce a variable dictionary with units, eligibility, response codes, transformations, and sources.
- Produce a cohort flow and missingness report without exposing participant records.
- Preserve immutable raw files locally and make all cleaning reproducible.
- Test duplicate-key rejection, missingness recoding, outcome unknown handling, and join row preservation using synthetic fixtures.

These are V0.2 acceptance criteria; the V0.1 catalogue does not implement ingestion.
