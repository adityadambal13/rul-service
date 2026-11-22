# syntax=docker/dockerfile:1

# ---------- builder: install production dependencies and the package into a venv ----------
FROM python:3.15-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1) Dependencies only. This layer is cached until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) The package itself, installed as a regular (non-editable) package.
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


# ---------- runtime: small image, no build tools, non-root user ----------
FROM python:3.15-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUL_MODEL_PATH=/models/rul_bundle.joblib

RUN useradd --create-home --uid 1000 appuser \
    && mkdir /models \
    && chown appuser /models

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

USER 1000:1000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]

CMD ["uvicorn", "rul_service.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ---------- bundled: the runtime image plus your trained model baked in ----------
# Always pass --target: `--target runtime` (model mounted at /models) or
# `--target bundled` (needs models/rul_bundle.joblib to exist when you build).
FROM runtime AS bundled
COPY --chown=appuser:appuser models/rul_bundle.joblib /models/rul_bundle.joblib