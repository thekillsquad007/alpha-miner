#!/usr/bin/env bash
# Source this before running HIP tools/miner on WSL2 with AMD GPU.
# Usage: source scripts/env-wsl-hip.sh
#
# WSL AMD needs:
#   /dev/dxg              (DXG bridge from Windows)
#   librocdxg.so          (ROCm DXG detection library)
#   HSA_ENABLE_DXG_DETECTION=1

# Prefer full toolkit (hipcc) from /opt/rocm, plus librocdxg if installed separately.
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export PATH="${ROCM_PATH}/bin:${PATH}"

# librocdxg often ships in a newer partial ROCm package (e.g. 7.2.3) while hipcc is on 7.2.0
_ROCDXG_CANDIDATES=(
  /opt/rocm-7.2.3/lib
  /opt/rocm/lib
  "${ROCM_PATH}/lib"
)
_EXTRA=""
for d in "${_ROCDXG_CANDIDATES[@]}"; do
  if [[ -e "$d/librocdxg.so" || -e "$d/librocdxg.so.1" ]]; then
    _EXTRA="${d}:${_EXTRA}"
  fi
done

export LD_LIBRARY_PATH="${_EXTRA}${ROCM_PATH}/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_ENABLE_DXG_DETECTION=1

# Helpful diagnostics (optional)
if [[ "${ALPHA_HIP_QUIET:-}" != "1" ]]; then
  echo "[env-wsl-hip] ROCM_PATH=$ROCM_PATH"
  echo "[env-wsl-hip] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  if [[ -e /dev/dxg ]]; then echo "[env-wsl-hip] /dev/dxg: present"; else echo "[env-wsl-hip] /dev/dxg: MISSING"; fi
  if [[ -e /dev/kfd ]]; then echo "[env-wsl-hip] /dev/kfd: present"; else echo "[env-wsl-hip] /dev/kfd: absent (normal on WSL)"; fi
  if command -v hipcc >/dev/null; then
    if python3 - <<'PY' 2>/dev/null
import ctypes, os
for p in os.environ.get("LD_LIBRARY_PATH","").split(":"):
  pass
try:
  ctypes.CDLL("librocdxg.so.1")
  print("[env-wsl-hip] librocdxg: OK")
except Exception as e:
  try:
    ctypes.CDLL("librocdxg.so")
    print("[env-wsl-hip] librocdxg: OK")
  except Exception as e2:
    print("[env-wsl-hip] librocdxg: FAIL", e2)
PY
    then :; fi
  fi
fi
