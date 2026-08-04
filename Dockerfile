# ============ Builder 阶段 ============
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install .

# ============ Runtime 阶段 ============
FROM python:3.11-slim AS runtime

# 非 root 用户（安全规范，见 AGENTS.md）
RUN useradd --create-home appuser
USER appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
