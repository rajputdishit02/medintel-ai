# Deployment

## Local application

```bash
python -m pip install -e .
medintel-app
```

The interface opens on `http://localhost:8501`. Run the acquisition, cohort, modelling, interpretation, and literature commands first to enable every interactive feature. Aggregate reports work directly from the repository.

## Container

```bash
docker compose up --build
```

The image runs as an unprivileged user and exposes a Streamlit health check. Participant data, trained model binaries, credentials, and the local PubMed corpus are excluded. PubMed evidence is acquired from NCBI on the first research-assistant question; the public risk explorer remains disabled unless an approved private model artifact is mounted.

Do not deploy this project as a clinical service. The model is internally evaluated on a cross-sectional survey outcome and has no clinical or external validation.
