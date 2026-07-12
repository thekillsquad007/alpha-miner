#!/usr/bin/env bash
# Build alpha-miner with NVIDIA CUDA multi-GPU support.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install CUDA toolkit or set PATH to /usr/local/cuda/bin" >&2
  exit 1
fi

echo "nvcc: $(nvcc --version | tail -1)"
nvidia-smi -L 2>/dev/null || true

cmake -S "$ROOT" -B "$ROOT/build-cuda" \
  -DCMAKE_BUILD_TYPE=Release \
  -DALPHA_MINER_CUDA=ON \
  -DALPHA_MINER_HIP=OFF \
  -DALPHA_MINER_OPENCL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"

cmake --build "$ROOT/build-cuda" -j"$(nproc)"

echo
echo "Binary: $ROOT/build-cuda/alpha-miner"
echo "  $ROOT/build-cuda/alpha-miner -l"
echo "  $ROOT/build-cuda/alpha-miner -o eu.rplant.xyz:7176 -u WALLET.rig -b cuda -d 0,1,2"
