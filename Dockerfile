# syntax=docker/dockerfile:1
#
# ResearchPilot image, used for the API, the Celery worker and the one-off migration job.
#
#   docker build -t researchpilot .
#
# * Python 3.11, CPU-only PyTorch (the default PyPI wheel bundles CUDA and is several GB).
# * Dependency versions are constrained by requirements/lock.txt, the versions the test suite
#   was run with; the first install layer only changes when dependencies change.
# * Models (~1.3 GB) are NOT baked in: they download on first use into /app/.cache/hf, which
#   docker-compose mounts as a volume so they persist across container restarts.
# * Runs as an unprivileged user; uvicorn/celery receive SIGTERM directly for graceful shutdown.

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------- builder stage
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv && pip install --upgrade pip setuptools wheel

WORKDIR /build
COPY pyproject.toml README.md ./
COPY requirements/lock.txt requirements/lock.txt

# Dependencies only (cached until pyproject.toml or the lock file change).
RUN grep -v -E '^\s*(#|--)' requirements/lock.txt > constraints.txt \
 && python - <<'EOF'
import tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
open("requirements.txt", "w").write("\n".join(deps) + "\n")
EOF
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "$(grep -E '^torch==' constraints.txt)" \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -c constraints.txt -r requirements.txt

# The application itself (no dependency resolution needed any more).
COPY app ./app
RUN pip install --no-deps .

# ---------------------------------------------------------------------------- runtime stage
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    LOG_JSON=true \
    HF_HOME=/app/.cache/hf \
    HF_HUB_DISABLE_TELEMETRY=1

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app app ./app
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app alembic.ini ./

# Writable locations: uploaded PDFs / SQLite fallback, model cache, evaluation results.
RUN mkdir -p /app/data/uploads /app/.cache/hf /app/experiments/results \
 && chown -R app:app /app/data /app/.cache /app/experiments

USER app
EXPOSE 8000

# Liveness only (no dependency checks), so a slow external service never restarts the API.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/live', timeout=4); sys.exit(0 if r.status == 200 else 1)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--timeout-graceful-shutdown", "30"]
