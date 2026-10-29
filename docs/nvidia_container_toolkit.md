# NVIDIA Container Toolkit (GPU training in Docker)

Only needed for the `ml-gpu` service and the dev container. The CPU service `ml` works without it.

Containers cannot see the GPU on their own. The toolkit mounts the host's driver and GPU
device into the container at start-up, so images stay driver-free and portable. Without it,
`docker compose run ml-gpu …` fails with:

```
failed to discover GPU vendor from CDI: no known GPU vendor found
```

Install it once on the host (official NVIDIA apt repository):

```bash
# 1. Signing key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 2. Package source
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 4. Register the runtime with Docker (writes /etc/docker/daemon.json) and restart Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Check that it worked. This should print your GPU:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

If you still see the CDI error, generate the GPU description manually and restart Docker:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
```

> ⚠️ **WSL2:** Never install a Linux NVIDIA driver inside WSL (`nvidia-driver-*`, or
> `cuda` meta-packages that pull one in). WSL gets the driver from Windows via
> `/usr/lib/wsl/lib`, and a Linux driver breaks that bridge. The Container Toolkit is
> not a driver and is safe to install.
>
> A local CUDA toolkit (`nvcc`) is **not** needed. PyTorch ships its own CUDA runtime,
> only the driver has to be new enough (CUDA 13.0 → driver ≥ 580).

Tested with: Ubuntu 24.04 on WSL2, Docker Engine 29.8, NVIDIA Container Toolkit 1.20.1,
Windows driver 596.08.

