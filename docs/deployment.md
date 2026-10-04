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

The image runs as an unprivileged user and exposes a Streamlit health check. Participant data, trained model binaries, credentials, and the local PubMed corpus are excluded from the image. Mount approved local artifacts at runtime if a private deployment needs interactive predictions or retrieval.

Do not deploy this project as a clinical service. The model is internally evaluated on a cross-sectional survey outcome and has no clinical or external validation.
