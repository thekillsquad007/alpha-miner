#!/usr/bin/env bash
# Mine ALPHA (blake3-an) on rplant — NVIDIA or AMD via OpenCL, else CPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/alpha-miner"
WALLET="${1:?usage: $0 <ALPHA_WALLET> [worker] [region]}"
WORKER="${2:-rig1}"
REGION="${3:-eu}"   # eu | na | asia
HOST="${REGION}.rplant.xyz"
PORT=7176

if [[ ! -x "$BIN" ]]; then
  echo "Build first: cmake -S $ROOT -B $ROOT/build -DCMAKE_BUILD_TYPE=Release && cmake --build $ROOT/build -j"
  exit 1
fi

# Prefer HIP build if present (AMD ROCm).
if [[ -x "$ROOT/build-hip/alpha-miner" ]]; then
  BIN="$ROOT/build-hip/alpha-miner"
fi
KERNEL="$ROOT/kernels/blake3_an.cl"
exec "$BIN" -o "${HOST}:${PORT}" -u "${WALLET}.${WORKER}" -p x -b auto -k "$KERNEL"
