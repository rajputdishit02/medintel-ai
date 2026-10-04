# Model card — MedIntel V0.5

## Model and intended use

The retained model is logistic regression with sigmoid probability calibration. It estimates association with a composite of self-reported prevalent cardiovascular disease history among NHANES August 2021–August 2023 adult participants.

It is an educational research artifact for demonstrating reproducible clinical data science. It must not be used for diagnosis, screening, treatment, patient counselling, or prediction of future events.

## Data and target

The modelling cohort contains 7,764 adults with a known composite outcome: 982 positive and 6,782 negative. A positive is any valid yes for reported heart failure, coronary heart disease, angina, heart attack, or stroke. All five responses must be valid no answers for a negative; 45 unknown outcomes were excluded.

Predictors are age, sex, race and Hispanic-origin category, income-to-poverty ratio, measured blood pressure, BMI, waist circumference, total and HDL cholesterol, HbA1c, and lifetime smoking status. Reported hypertension/cholesterol diagnoses and treatment variables are excluded due to healthcare-contact, treatment, and reverse-causation concerns.

## Evaluation

A fixed stratified 80/20 split produced 6,211 training and 1,553 test records. Calibration choice and the exploratory threshold used training data only. On the test partition, the calibrated model achieved ROC AUC 0.793, average precision 0.390, Brier score 0.093, and log loss 0.316.

The exploratory threshold of 0.121 targeted at least 80% sensitivity in out-of-fold training predictions. Test sensitivity was 0.770, specificity 0.690, positive predictive value 0.264, and negative predictive value 0.954. This threshold is not clinically selected or validated.

## Subgroup observations

Sex-group ROC AUC was 0.796 for men and 0.786 for women. The age audit exposed a material limitation: the 20–39 test group contained only 12 positive cases, ROC AUC was 0.485, and sensitivity at the exploratory threshold was zero. Subgroup results are unweighted, internally evaluated, and too limited to establish fairness or transportability.

## Explanation scope

Global SHAP values are computed for the underlying uncalibrated logistic model on the log-odds scale. Age has the largest mean absolute attribution. These values explain model behaviour relative to the selected training background; they do not identify causes, biological importance, or treatment effects. Correlated variables and missingness indicators can redistribute attribution.

## Limitations

- Cross-sectional self-report cannot support future-risk claims.
- The composite target combines clinically different conditions and is not adjudicated.
- Measurements may follow diagnosis or treatment, creating reverse causation.
- Missing values are imputed and missingness can itself influence the model.
- Evaluation is internal, unweighted, and from one survey cycle.
- No external, temporal, prospective, or clinical validation exists.
- Predictive values will change with outcome prevalence.

The model artifact is stored locally and excluded from Git. Recreate it with `python -m medintel baseline --data-dir data` followed by `python -m medintel interpret --data-dir data`.
