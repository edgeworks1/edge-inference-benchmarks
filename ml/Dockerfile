# torch-Variante beim Build wählbar: cpu (CI, Default) oder cu130 (lokales GPU-Training).
# Globales ARG vor dem ersten FROM; jede Stage, die es braucht, holt es mit `ARG TORCH_VARIANT`.
ARG TORCH_VARIANT=cpu

# =============================================================================
# Stage "base": Python + uv + alle Abhängigkeiten aus uv.lock
# Gemeinsamer Unterbau für runtime (CI) und später dev (Dev Container).
# =============================================================================
FROM python:3.11-slim-trixie AS base

# uv als einzelnes Binary aus dem offiziellen Image holen (gleiche Version wie lokal)
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv

WORKDIR /app

# UV_COMPILE_BYTECODE: .pyc schon beim Build erzeugen -> schnellerer Start
# UV_LINK_MODE=copy:   Pakete kopieren statt verlinken (der Cache liegt im Cache-Mount,
#                      Hardlinks dorthin wären nach dem Build ins Leere zeigend)
# UV_PYTHON_DOWNLOADS: das Python aus dem Basis-Image nutzen, keins nachladen
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# ARGs gelten nur in ihrer Stage -> hier erneut holen (für das uv sync unten)
ARG TORCH_VARIANT

# ---- Abhängigkeiten: nur die Lock-Dateien, noch kein Code ----
COPY pyproject.toml uv.lock .python-version ./
# Cache-Mount: Speichert heruntergeladene .whl auf dem Host (spart ~2 GB erneuter Download
# im Image, verhindert erneutes Laden aus dem Netz bei Paket-Änderungen).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project --extra ${TORCH_VARIANT}

# Programme der venv (python, ble-cs) ohne Pfadangabe aufrufbar machen
# Statt '/app/.venv/bin/ble-cs' reicht jetzt einfach 'ble-cs' oder 'python'.
ENV PATH="/app/.venv/bin:$PATH"

# Non-Root-User mit Host-UID/GID (verhindert Root-Rechte in Volumes).
# Hier unten platziert, damit abweichende CI-UIDs den schweren uv-Cache oben nicht invalidieren.
ARG UID=1000
ARG GID=1000
# --shell /bin/bash: Debian-Default ist /bin/sh (dash) -> keine Pfeiltasten-Historie, keine .bashrc
RUN groupadd --gid ${GID} app && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash app

# =============================================================================
# Stage "runtime": fertiges Programm (CI, `docker compose run ml / ml-gpu`)
# =============================================================================
FROM base AS runtime

ARG TORCH_VARIANT

# ---- Code: ändert sich oft, deshalb erst NACH den Abhängigkeiten ----
# Eine Code-Änderung baut nur die Layer ab hier neu, der Install-Layer oben bleibt gecacht.
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --extra ${TORCH_VARIANT}

# Nicht als root laufen (User wurde in base angelegt)
USER app

# Kapselt das Image als CLI-Tool: Argumente an "docker run" werden direkt an "ble-cs" durchgereicht (Default: --help).
ENTRYPOINT ["ble-cs"]
CMD ["--help"]

# =============================================================================
# Stage "dev": Arbeitsumgebung für den Dev Container (VS Code)
# Das Repo wird zur Laufzeit unter /workspace eingehängt, NICHT unter /app.
# Dadurch bleibt die venv aus base (/app/.venv) unberührt und wird weiterbenutzt.
# =============================================================================
FROM base AS dev
ARG TORCH_VARIANT

# git für Commits aus dem Container (VS Code reicht .gitconfig und Credentials durch)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# uv soll die venv aus base benutzen, nicht /workspace/ml/.venv (das ist die Host-venv)
# UV_NO_SYNC: die venv gehört root und ist schreibgeschützt -> `uv run` soll nicht
# synchronisieren. Paketänderungen: `uv add --no-sync <paket>`, dann Container neu bauen.
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_NO_SYNC=1

# Projekt dort ablegen, wo später der Mount liegt -> die editable Installation zeigt auf
# /workspace/ml/src. Zur Laufzeit überdeckt der Mount diese Kopie mit dem echten Repo.
WORKDIR /workspace/ml
COPY pyproject.toml uv.lock .python-version ./
COPY src/ src/

# Dev-Pakete (pytest, ruff, ipykernel ...) + dein Paket editable dazu; der Rest ist schon da
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra ${TORCH_VARIANT}

# Optionaler VS-Code-Komfort: Hält Shell-Historie und Caches über Container-Rebuilds hinweg persistent
RUN mkdir -p /commandhistory /home/app/.cache \
    && touch /commandhistory/.bash_history \
    && chown -R app:app /commandhistory /home/app/.cache \
    && echo 'export HISTFILE=/commandhistory/.bash_history PROMPT_COMMAND="history -a"' >> /home/app/.bashrc

USER app
