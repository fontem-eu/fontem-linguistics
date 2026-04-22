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
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --index-url https://nexus.void42.internal/repository/pytorch-cpu/simple/ \
        --extra-index-url https://nexus.void42.internal/repository/pypi-proxy/simple/ \
        -r requirements-ml.txt || pip install --no-cache-dir -r requirements-ml.txt

COPY src/ src/

USER appuser

CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
