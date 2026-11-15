# torch build chosen at build time: cpu (CI, default) or cu130 (local GPU training).
# Global ARG before the first FROM; every stage that needs it re-declares `ARG TORCH_VARIANT`.
ARG TORCH_VARIANT=cpu

# Stage "base": Python + uv + all dependencies from uv.lock
# Shared foundation of runtime (pipeline, CI) and dev (dev container).
FROM python:3.11-slim-trixie AS base

# uv as a single binary from its official image (same version as used locally)
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

WORKDIR /app

# UV_COMPILE_BYTECODE: create .pyc files at build time -> faster start
# UV_LINK_MODE=copy:   copy packages instead of linking (the uv cache is a cache mount,
#                      hard links into it would point nowhere after the build)
# UV_PYTHON_DOWNLOADS: use the Python of the base image, never download another one
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# ARGs are scoped to their stage -> declare again for the uv sync below
ARG TORCH_VARIANT

# ---- Dependencies: lock files only, no code yet ----
COPY pyproject.toml uv.lock .python-version ./
# Cache mount: downloaded wheels stay on the host between builds (about 2 GB for torch),
# so a dependency change does not download everything again.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project --extra ${TORCH_VARIANT}

# Make the venv's programs (python, edge-ib) callable without a path
ENV PATH="/app/.venv/bin:$PATH"

# Non-root user with the host's UID/GID, so files written to mounted folders belong to you.
# Placed after the dependency install: a different UID (e.g. in CI) keeps the heavy layer above cached.
ARG UID=1000
ARG GID=1000
# --shell /bin/bash: the Debian default /bin/sh (dash) has no history and ignores .bashrc
RUN groupadd --gid ${GID} app && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash app

# Stage "runtime": the finished program (pipeline, CI, `docker compose run ml / ml-gpu`)
FROM base AS runtime

ARG TORCH_VARIANT

# ---- Code: changes often, so it is copied AFTER the dependencies ----
# A code change only rebuilds the layers from here on, the install layer above stays cached.
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --extra ${TORCH_VARIANT}

# Do not run as root (the user was created in base)
USER app

# The image behaves like the CLI: arguments of `docker run` go straight to edge-ib (default: --help)
ENTRYPOINT ["edge-ib"]
CMD ["--help"]

# Stage "dev": working environment of the VS Code dev container
# The repo is mounted at /workspace at runtime, NOT at /app, so the venv from base
# (/app/.venv) stays untouched and is reused.
FROM base AS dev
ARG TORCH_VARIANT

# git for commits from inside the container (VS Code forwards .gitconfig and credentials),
# make for the example targets in ml/experiments/
RUN apt-get update \
    && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/*

# uv must use the venv from base, not /workspace/ml/.venv (that one belongs to the host)
# UV_NO_SYNC: the venv belongs to root and is read-only, so `uv run` must not sync it.
# Adding a package: `uv add --no-sync <package>`, then rebuild the container.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_NO_SYNC=1

# Copy the project to where the mount will be -> the editable install points to
# /workspace/ml/src. At runtime the mount replaces this copy with the real repo.
WORKDIR /workspace/ml
COPY pyproject.toml uv.lock .python-version ./
COPY src/ src/

# Dev packages (pytest, ruff, ipykernel ...) plus this package as editable; the rest is already installed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra ${TORCH_VARIANT}

# Keep shell history and caches across container rebuilds (named volumes, see devcontainer.json)
RUN mkdir -p /commandhistory /home/app/.cache \
    && touch /commandhistory/.bash_history \
    && chown -R app:app /commandhistory /home/app/.cache \
    && echo 'export HISTFILE=/commandhistory/.bash_history PROMPT_COMMAND="history -a"' >> /home/app/.bashrc

USER app
