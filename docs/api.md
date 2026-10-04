# V0.6 research API

The FastAPI service exposes the locally rebuilt V0.5 model through a strict, versioned contract. It accepts research feature records only. It does not accept names, contact details, free text, clinical notes, or other patient identifiers.

## Run locally

Rebuild the ignored model artifact first:

```bash
python -m medintel build-cohort --data-dir data
python -m medintel baseline --data-dir data
python -m medintel interpret --data-dir data
medintel-api
```

Open `http://127.0.0.1:8000/docs` for the generated interactive schema. Bind to loopback only for local development. A different model path can be supplied through `MEDINTEL_MODEL_PATH`.

## Endpoints

- `GET /healthz` reports application and model readiness without exposing local paths.
- `GET /v1/model` returns target, calibration, threshold, held-out metrics, and usage restrictions.
- `POST /v1/predictions` validates one feature record and returns a probability plus the exploratory threshold comparison.

Example request:

```json
{
  "age_years": 58,
  "sex": "female",
  "race_ethnicity": "non_hispanic_asian",
  "income_poverty_ratio": 2.5,
  "systolic_bp": 132,
  "diastolic_bp": 82,
  "bmi": 29.4,
  "waist_cm": 96,
  "total_cholesterol_mg_dl": 205,
  "hdl_cholesterol_mg_dl": 48,
  "hba1c_percent": 5.9,
  "ever_smoked": false
}
```

The returned probability describes association with prevalent self-reported history in this research cohort. It is not future risk, a diagnosis, or medical advice. The threshold is an exploratory analysis choice and has not been clinically validated.
