FROM python:3.12-slim

COPY void42-ca.crt /usr/local/share/ca-certificates/void42-ca.crt
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    update-ca-certificates && rm -rf /var/lib/apt/lists/*

ENV PIP_INDEX_URL=https://nexus.void42.internal/repository/pypi-proxy/simple/ \
    PIP_TRUSTED_HOST=nexus.void42.internal \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models

RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /models && chown appuser:appuser /models

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ML backends (nllb-local + labse-local). The torch wheel on PyPI ships with
# the bundled CUDA runtime (~1.5 GB of nvidia-* deps) even though our pods
# are CPU-only — functionally harmless, just image bloat. To shrink later:
# configure a Nexus *raw* proxy for https://download.pytorch.org/whl/cpu/
# (the pypi-format proxy doesn't understand that upstream's layout) and
# switch `--index-url` back to it.
RUN pip install --no-cache-dir --prefer-binary -r requirements-ml.txt

COPY src/ src/

USER appuser

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
