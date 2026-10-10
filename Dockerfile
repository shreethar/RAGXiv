# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.8.10@sha256:b05b3d61eb2b264ed785265b71155738a0d3d382ea0699e048d4b36f90b88788 AS uv

FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f AS builder

ENV UV_LINK_MODE=copy

WORKDIR /app

COPY --from=uv /uv /bin/uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN addgroup --system ragxiv \
    && adduser --system --ingroup ragxiv --no-create-home ragxiv

COPY --from=builder --chown=ragxiv:ragxiv /app/.venv ./.venv
COPY --chown=ragxiv:ragxiv app ./app
COPY --chown=ragxiv:ragxiv alembic ./alembic
COPY --chown=ragxiv:ragxiv alembic.ini ./alembic.ini

USER ragxiv

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
