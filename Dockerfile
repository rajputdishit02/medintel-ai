FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN groupadd --system medintel && useradd --system --gid medintel --create-home medintel
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY --chown=medintel:medintel .streamlit ./.streamlit
COPY --chown=medintel:medintel streamlit_app.py ./
COPY --chown=medintel:medintel reports ./reports
COPY --chown=medintel:medintel data/metadata ./data/metadata
COPY --chown=medintel:medintel docs ./docs

USER medintel
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
