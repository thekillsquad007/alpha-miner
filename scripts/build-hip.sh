#!/usr/bin/env bash
# Build alpha-miner with AMD HIP (ROCm hipcc), including WSL2 DXG setup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT/scripts/env-wsl-hip.sh"

if ! command -v hipcc >/dev/null 2>&1; then
  echo "hipcc not found. Install ROCm or set ROCM_PATH." >&2
  exit 1
fi

echo "Using hipcc: $(command -v hipcc)"
hipcc --version 2>&1 | head -4

# Detect GPU arch via hipGetDeviceCount (reliable). Avoid bare `offload-arch` on
# WSL — it can spin at 100% CPU when DXG setup is partial.
OFFLOAD_ARCH="${HIP_OFFLOAD_ARCH:-}"
cat > /tmp/alpha_hip_detect.cpp <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n <= 0) {
    fprintf(stderr, "No HIP devices visible. On WSL: source scripts/env-wsl-hip.sh\n");
    fprintf(stderr, "Need: /dev/dxg + librocdxg + HSA_ENABLE_DXG_DETECTION=1\n");
    return 1;
  }
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t p{};
    hipGetDeviceProperties(&p, i);
    printf("HIP device %d: %s (%s) cu=%d mem=%.0fMiB\n", i, p.name, p.gcnArchName,
           p.multiProcessorCount, p.totalGlobalMem / (1024.0 * 1024.0));
    if (i == 0 && p.gcnArchName[0]) printf("%s\n", p.gcnArchName);
  }
  return 0;
}
EOF
hipcc -O2 /tmp/alpha_hip_detect.cpp -o /tmp/alpha_hip_detect
DETECT_OUT="$(/tmp/alpha_hip_detect)"
echo "$DETECT_OUT"
if [[ -z "$OFFLOAD_ARCH" ]]; then
  OFFLOAD_ARCH="$(echo "$DETECT_OUT" | grep -E '^gfx[0-9a-f]+$' | head -1 || true)"
fi
if [[ -n "$OFFLOAD_ARCH" ]]; then
  echo "GPU offload arch: $OFFLOAD_ARCH"
else
  echo "WARN: could not detect GPU arch; hipcc default may be used"
fi

cmake -S "$ROOT" -B "$ROOT/build-hip" \
  -DCMAKE_BUILD_TYPE=Release \
  -DALPHA_MINER_HIP=ON \
  -DALPHA_MINER_OPENCL=OFF \
  -DALPHA_MINER_CUDA=OFF \
  ${OFFLOAD_ARCH:+-DHIP_OFFLOAD_ARCH=${OFFLOAD_ARCH}}

cmake --build "$ROOT/build-hip" -j"$(nproc)"

# Package a launcher that sets WSL DXG env
mkdir -p "$ROOT/dist/wsl-hip"
cp -f "$ROOT/build-hip/alpha-miner" "$ROOT/dist/wsl-hip/alpha-miner"
cat > "$ROOT/dist/wsl-hip/run-hip.sh" <<EOF
#!/usr/bin/env bash
ROOT="\$(cd "\$(dirname "\$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/../scripts/env-wsl-hip.sh" 2>/dev/null || source "$ROOT/../../scripts/env-wsl-hip.sh"
export ALPHA_HIP_QUIET=1
exec "\$ROOT/alpha-miner" "\$@"
EOF
# fix paths in launcher
cat > "$ROOT/dist/wsl-hip/run-hip.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/../.." 2>/dev/null && pwd)"
if [[ -f "$ROOT/../scripts/env-wsl-hip.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/../scripts/env-wsl-hip.sh"
elif [[ -f /mnt/e/Business/alpha-miner/scripts/env-wsl-hip.sh ]]; then
  # shellcheck disable=SC1091
  source /mnt/e/Business/alpha-miner/scripts/env-wsl-hip.sh
fi
export ALPHA_HIP_QUIET=1
exec "$ROOT/alpha-miner" "$@"
EOF
chmod +x "$ROOT/dist/wsl-hip/run-hip.sh"

echo
echo "Binary: $ROOT/build-hip/alpha-miner"
echo "WSL package: $ROOT/dist/wsl-hip/"
echo "Run:"
echo "  source $ROOT/scripts/env-wsl-hip.sh"
echo "  $ROOT/build-hip/alpha-miner -l"
echo "  $ROOT/build-hip/alpha-miner -o eu.rplant.xyz:7176 -u WALLET.worker -b hip"
