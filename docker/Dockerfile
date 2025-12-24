# /****************************************************************************
# *  @Copyright 2025 XtalPi Systems.
# *  @Author [tongfu.e@xtalpi.com].
# *  @Date [2025-12-24 13:22:42].
# *  @Description Dockerfile for molx_agent.
# *  @FIXME: 暂不可用
# ***************************************************************************/

FROM python:3.12-slim-bookworm

ENV LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  UV_LINK_MODE=copy \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV=false
RUN if [ "$INSTALL_DEV" = "true" ]; then \
  uv sync --frozen --extra dev; \
  else \
  uv sync --frozen; \
  fi

# Copy the rest of the application
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
  chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD uv run molx-agent --version || exit 1

CMD ["uv", "run", "molx-agent"]
