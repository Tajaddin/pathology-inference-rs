# Python baseline: FastAPI + onnxruntime with the same /v1/classify contract (for the comparison only).
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY training/pyproject.toml training/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --extra baseline --no-install-project
COPY training/pcam ./pcam
COPY training/baseline ./baseline
RUN useradd --create-home --uid 10001 app
USER app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080
CMD ["uvicorn", "baseline.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4", "--no-access-log"]
