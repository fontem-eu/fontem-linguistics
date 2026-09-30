# fontem-linguistics: embeddings, translation and language detection.
#
# Built like the other Python services: a venv made in Chainguard's -dev
# image, copied into the distroless runtime (no shell, no package manager),
# pip removed. The runtime base is pinned by digest (Renovate keeps it
# current), so the Python version only moves when the digest does.
FROM cgr.void42.internal/chainguard/python:latest-dev@sha256:5eef76bbb8d9f815317da126075705202b8ca5c2a151d723e7ecdf0373d9d861 AS build
USER root
ENV PIP_INDEX_URL=https://nexus.void42.internal/repository/pypi-proxy/simple/ \
    PIP_TRUSTED_HOST=nexus.void42.internal
COPY void42-ca.crt /tmp/void42-ca.crt
RUN cat /tmp/void42-ca.crt >> /etc/ssl/certs/ca-certificates.crt
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# CPU-only torch: the service requests no GPU, and PyPI's default torch wheel
# depends on 15 NVIDIA CUDA packages (the image was 6.3 GB). Only torch comes
# from the PyTorch index (--no-deps); its dependencies and everything else
# resolve from the Nexus PyPI proxy in the next step.
ARG TORCH_VERSION=2.14.0
RUN pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}+cpu"
RUN pip install --no-cache-dir --prefer-binary -r requirements-ml.txt \
 && pip check \
 && ! pip list --format=freeze | grep -i -E '^nvidia-' \
 && pip uninstall -y pip \
 && mkdir /models

FROM cgr.void42.internal/chainguard/python:latest@sha256:a1775c7276078865461ee5714954284f12809f333433d856d720b249c65c11b2
WORKDIR /app
COPY --from=build /venv /venv
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
# Where the fetch-models init container puts the models (an emptyDir in the
# cluster); owned by the service user so it works on a plain volume too.
COPY --from=build --chown=65532:65532 /models /models
ENV PATH="/venv/bin:$PATH" \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models
COPY src/ src/
USER 65532
EXPOSE 8080
ENTRYPOINT ["/venv/bin/python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
