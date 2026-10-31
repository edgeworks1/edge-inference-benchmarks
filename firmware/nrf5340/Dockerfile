# =============================================================================
# Build environment for the nRF5340 firmware (Zephyr RTOS / nRF Connect SDK v2.6.0)
# Tools and SDK live in the image; the application code is mounted as a volume.
# =============================================================================
FROM python:3.11-slim-trixie AS build

# Native host tools for Zephyr (CMake, Ninja generator, devicetree and Kconfig)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake ninja-build device-tree-compiler gperf git wget xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Zephyr SDK 0.16.5: minimal bundle plus only the ARM Cortex-M cross compiler (sha256-verified)
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

# Python package manager (uv) and a dedicated venv for west
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

ARG WEST_VERSION=1.5.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /ncs/.venv \
    && uv pip install --python /ncs/.venv west==${WEST_VERSION}

ENV PATH="/ncs/.venv/bin:$PATH"

# Initialize the NCS workspace (a shallow clone saves disk space and build time)
WORKDIR /ncs
COPY west.yml manifest/west.yml
RUN west init -l manifest \
    && west update --narrow -o=--depth=1 \
    && west config zephyr.base zephyr

ENV ZEPHYR_BASE=/ncs/zephyr

# Python dependencies of the SDK repos (Kconfiglib, pyelftools ...)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /ncs/.venv \
        -r zephyr/scripts/requirements.txt \
        -r nrf/scripts/requirements.txt \
        -r bootloader/mcuboot/scripts/requirements.txt

# Non-root setup: UID/GID 1000 avoids permission problems with mounted host files.
# safe.directory: git inside the container accepts the SDK repos owned by root.
RUN git config --system --add safe.directory '/ncs/*'

ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} app \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash app \
    && install -d -o app -g app /home/app/.cache

WORKDIR /app
USER app

# =============================================================================
# Stage "renode": nRF5340 simulation (independent of SDK and compiler -> small image)
# Gets the built ELF through the mount (build/), not from the build stage.
# =============================================================================
FROM debian:trixie-slim AS renode

# Portable dotnet variant (ships its own .NET runtime), sha256-verified
# libicu76: Unicode library that .NET needs at runtime (not included in the package)
# wget is only needed for the download and removed afterwards
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
