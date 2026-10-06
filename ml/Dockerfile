# ---- Basis: schlankes Debian 13 mit Python 3.11 ----
FROM python:3.11-slim-trixie

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

# torch-Variante beim Build wählbar: cpu (CI, Default) oder cu130 (lokales GPU-Training)
ARG TORCH_VARIANT=cpu

# ---- Abhängigkeiten: nur die Lock-Dateien, noch kein Code ----
COPY pyproject.toml uv.lock .python-version ./
# Cache-Mount: uvs Download-Cache liegt außerhalb des Images (spart ~2 GB im Layer)
# und bleibt zwischen Builds erhalten (kein erneuter Download)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project --extra ${TORCH_VARIANT}

# ---- Code: ändert sich oft, deshalb erst NACH den Abhängigkeiten ----
# Eine Code-Änderung baut nur die Layer ab hier neu, der Install-Layer oben bleibt gecacht.
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --extra ${TORCH_VARIANT}

# Programme der venv (python, ble-cs) ohne Pfadangabe aufrufbar machen
ENV PATH="/app/.venv/bin:$PATH"

# ---- Nicht als root laufen ----
# Echter User (Name + Home) mit derselben UID/GID wie der Host-User:
# Dateien in eingehängten Ordnern gehören dann dir statt root.
# Default 1000 = dein WSL-User; in der CI per --build-arg UID=$(id -u) anpassbar.
# Ist erst hier unten, damit Layer mit den Abhängigkeiten nicht für jeden User 
#neu gebaut werden muss. (also z.b. wenn GitLab baut, müsste jedes nachfolgende Layer neu gebaut werden, weil der User anders ist)
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} app && useradd --uid ${UID} --gid ${GID} --create-home app
USER app

# `docker run <image> <args>` wird zu `ble-cs <args>`; ohne Argumente: Hilfe
ENTRYPOINT ["ble-cs"]
CMD ["--help"]
