# V0.3 modelling protocol

## Intended analysis

The target is prevalent, self-reported cardiovascular disease history: any valid yes across reported congestive heart failure, coronary heart disease, angina, heart attack, or stroke. A negative requires valid no responses to all five questions. Unknown outcomes are excluded rather than relabelled.

This is an association model for research. It does not predict a future event, establish a diagnosis, or estimate the effect of changing a risk factor. Because measurements were collected after or alongside reported diagnoses, reverse causation and treatment effects are plausible.

## Frozen V0.3 predictors

The baseline uses age, sex, race and Hispanic-origin category, family income-to-poverty ratio, measured mean systolic and diastolic blood pressure, BMI, waist circumference, total cholesterol, HDL cholesterol, HbA1c, and lifetime smoking status.

The following are excluded:

- `SEQN`, weights, strata, and PSU identifiers: identifiers or survey-design fields, not participant predictors.
- `MCQ160B-F` and `cvd_history`: source target items or the derived target.
- `BPQ020`, `BPQ080`, `BPQ150`, and `BPQ101D`: reported hypertension/cholesterol diagnoses and treatment can encode healthcare contact or post-diagnosis treatment.
- Current smoking frequency: structurally missing for never-smokers and unnecessary for the first baseline.
- Risk-factor threshold derivatives: redundant transforms of included continuous measurements.

The feature set is frozen before test-set evaluation. Later experiments must not silently change this baseline.

## Evaluation contract

Use a fixed, stratified 80/20 split with seed 417. Fit all imputation, scaling, and encoding inside each training pipeline. Compare a class-prior dummy model with logistic regression. Report ROC AUC, average precision, Brier score, log loss, and threshold-0.5 sensitivity and specificity on the untouched test partition.

V0.3 evaluation is unweighted and describes discrimination among sampled participants. Survey weights remain reserved for population inference in V0.2. The test set supports internal comparison only; there is no temporal or external validation. Calibration analysis, threshold selection, subgroup evaluation, and uncertainty intervals belong to later milestones.
