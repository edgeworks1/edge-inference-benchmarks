# =============================================================================
# Build-Umgebung für nRF5340-Firmware (Zephyr RTOS / nRF Connect SDK v2.6.0)
# Werkzeuge und SDK liegen im Image; Applikations-Code wird per Volume gemountet.
# =============================================================================
FROM python:3.11-slim-trixie AS build

# Native Host-Tools für Zephyr (CMake, Ninja-Generator, Devicetree & Kconfig)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake ninja-build device-tree-compiler gperf git wget xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Zephyr SDK 0.16.5: Minimal-Bundle + ausschließlich ARM-Cortex-M Cross-Compiler
ARG ZEPHYR_SDK_VERSION=0.16.5
ARG SDK_URL=https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${ZEPHYR_SDK_VERSION}
RUN cd /tmp \
    && wget -q ${SDK_URL}/zephyr-sdk-${ZEPHYR_SDK_VERSION}_linux-x86_64_minimal.tar.xz \
    && wget -q ${SDK_URL}/toolchain_linux-x86_64_arm-zephyr-eabi.tar.xz \
    && echo "933ab6b4cd78dccf8fe0cf59d1b626efc309008408eaf848173c70360f848ee1  zephyr-sdk-${ZEPHYR_SDK_VERSION}_linux-x86_64_minimal.tar.xz" | sha256sum -c - \
    && echo "d4278e95af74f8e9210b417e1cdc3ed45936cf6db9685d4174000a1a95b9f095  toolchain_linux-x86_64_arm-zephyr-eabi.tar.xz" | sha256sum -c - \
    && tar -xf zephyr-sdk-${ZEPHYR_SDK_VERSION}_linux-x86_64_minimal.tar.xz -C /opt \
    && tar -xf toolchain_linux-x86_64_arm-zephyr-eabi.tar.xz -C /opt/zephyr-sdk-${ZEPHYR_SDK_VERSION} \
    && rm /tmp/*.tar.xz

ENV ZEPHYR_SDK_INSTALL_DIR=/opt/zephyr-sdk-${ZEPHYR_SDK_VERSION}

# Python-Paketmanager (uv) & dedizierte venv für west
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

ARG WEST_VERSION=1.5.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /ncs/.venv \
    && uv pip install --python /ncs/.venv west==${WEST_VERSION}

ENV PATH="/ncs/.venv/bin:$PATH"

# NCS Workspace initialisieren (Shallow Clone spart Speicherplatz und Build-Zeit)
WORKDIR /ncs
COPY west.yml manifest/west.yml
RUN west init -l manifest \
    && west update --narrow -o=--depth=1 \
    && west config zephyr.base zephyr

ENV ZEPHYR_BASE=/ncs/zephyr

# Python-Abhängigkeiten aus den SDK-Repos (Kconfiglib, pyelftools etc.)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /ncs/.venv \
        -r zephyr/scripts/requirements.txt \
        -r nrf/scripts/requirements.txt \
        -r bootloader/mcuboot/scripts/requirements.txt

# Non-Root-Setup: UID/GID 1000 verhindert Rechteprobleme mit gemounteten Host-Dateien
RUN git config --system --add safe.directory '/ncs/*'

ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} app \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash app \
    && install -d -o app -g app /home/app/.cache

WORKDIR /app
USER app

# =============================================================================
# Stage "renode": Simulation des nRF5340 (unabhängig von SDK/Compiler -> schlankes Image)
# Bekommt das fertige ELF über den Mount (build/), nicht aus der Build-Stage.
# =============================================================================
FROM debian:trixie-slim AS renode

# Portable-dotnet-Variante = gleiche wie lokal (bringt eigene .NET-Laufzeit mit)
# libicu76: Unicode-Bibliothek, die .NET zur Laufzeit braucht (nicht im Paket enthalten)
# wget/ca-certificates nur für den Download, danach wieder entfernt
ARG RENODE_VERSION=1.16.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libicu76 ca-certificates wget \
    && wget -q https://github.com/renode/renode/releases/download/v${RENODE_VERSION}/renode-${RENODE_VERSION}.linux-portable-dotnet.tar.gz -O /tmp/renode.tar.gz \
    && echo "00e113cdbd0f5354cf2f64bbe3f5a070d8958409542fca66e45ac97d982938c0  /tmp/renode.tar.gz" | sha256sum -c - \
    && mkdir /opt/renode \
    && tar -xzf /tmp/renode.tar.gz -C /opt/renode --strip-components=1 \
    && rm /tmp/renode.tar.gz \
    && apt-get purge -y wget && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/renode:$PATH"

ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} app \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash app

WORKDIR /app
USER app
