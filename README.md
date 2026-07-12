# alpha-miner

Open-source **Alphanumeric (ALPHA)** pool miner for **NVIDIA and AMD** GPUs.

**Developer fee: 2%** (second stratum session to the built-in fee wallet).

| Item | Value |
|------|--------|
| Algorithm | `blake3-an` — single BLAKE3 over the **92-byte** block header |
| Consensus source | [OSXBasedAnon/alphanumeric `gpu-mining`](https://github.com/OSXBasedAnon/alphanumeric/tree/gpu-mining) |
| Pool protocol | Monero-style JSON-RPC (`login` / `job` / `submit` / `keepalived`) |
| Reference pool | `stratum+tcp://eu.rplant.xyz:7176` (also `na` / `asia`) |
| Example multipool UI | [multipooldd.com/dashboard](https://multipooldd.com/dashboard) (RandomX coins today; same dashboard class of product) |

## Backends

| Backend | Hardware | Build flag |
|---------|----------|------------|
| **HIP** | **AMD Radeon (ROCm)** — preferred | `-DALPHA_MINER_HIP=ON` (default if `hipcc` found) |
| **OpenCL** | AMD + NVIDIA | `-DALPHA_MINER_OPENCL=ON` |
| **CPU** | any x86_64 | always |
| **CUDA** | NVIDIA | optional; or use [gpuminer-rplant-cuda](https://github.com/rplant-pool/gpuminer-rplant/releases) |

## Quick start

### Build AMD HIP miner (ROCm / WSL)

```bash
cd alpha-miner
# WSL2: loads librocdxg + HSA_ENABLE_DXG_DETECTION for /dev/dxg
source scripts/env-wsl-hip.sh

# Detect GPU (should print RX / gfx####)
hipcc -O2 -x hip - <<'EOF' -o /tmp/hip_det && /tmp/hip_det
#include <hip/hip_runtime.h>
#include <cstdio>
int main(){int n=0; hipGetDeviceCount(&n); printf("devices=%d\n",n);
  for(int i=0;i<n;i++){hipDeviceProp_t p{}; hipGetDeviceProperties(&p,i);
  printf("%d: %s %s\n",i,p.name,p.gcnArchName);} }
EOF

bash scripts/build-hip.sh
# → build-hip/alpha-miner  and  dist/wsl-hip/
```

**WSL note:** `/dev/kfd` is usually absent; the GPU is exposed as `/dev/dxg`. You need:
- `HSA_ENABLE_DXG_DETECTION=1`
- `librocdxg` on `LD_LIBRARY_PATH` (e.g. `/opt/rocm-7.2.3/lib` if hipcc is on 7.2.0)

Manual cmake (after `source scripts/env-wsl-hip.sh`):

```bash
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DALPHA_MINER_HIP=ON -DALPHA_MINER_OPENCL=OFF \
  -DHIP_OFFLOAD_ARCH=gfx1101   # from hipGetDeviceProperties
cmake --build build-hip -j$(nproc)
```

### Build OpenCL / CPU (no ROCm)

```bash
sudo apt install -y build-essential cmake ocl-icd-opencl-dev opencl-headers
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DALPHA_MINER_HIP=OFF
cmake --build build -j$(nproc)
```

### Mine on rplant (ALPHA port 7176)

```bash
# AMD HIP (recommended on Radeon)
./build-hip/alpha-miner -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b hip

# Auto: HIP → OpenCL → CPU
./build-hip/alpha-miner -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1

# OpenCL (AMD or NVIDIA)
./build/alpha-miner -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b opencl -k kernels/blake3_an.cl

# CPU only
./build/alpha-miner -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b cpu -t 16

# List GPUs
./build-hip/alpha-miner -l
```

### Windows / HiveOS production binaries

Prebuilt high-performance binaries from the rplant team (CUDA + OpenCL):

- https://github.com/rplant-pool/gpuminer-rplant/releases

```text
gpuminer-rplant-cuda   -a blake3-an -o 7176 -u WALLET.worker -p x
gpuminer-rplant-opencl -a blake3-an -o 7176 -u WALLET.worker -p x
```

## Header / share format

Matches `alphanumeric` `Block::calculate_hash_for_block` and `gpu_miner::build_header`:

```
offset  size  field
0       4     index (u32 LE)
4       32    previous_hash
36      8     timestamp (u64 LE)
44      8     nonce (u64 LE)   ← miner searches low 48 bits; pool binds high 16 (extranonce)
52      8     difficulty (u64 LE)
60      32    merkle_root
= 92 bytes → BLAKE3 → 32-byte hash
```

Share is valid when `hash <= target` (big-endian byte compare). Submit:

```json
{"method":"submit","params":{"id":"<session>","job_id":"...","nonce":"<16 hex LE>","result":"<64 hex hash>"}}
```

## Solo GPU mining (node-integrated)

The upstream node already includes a WGSL/wgpu GPU path:

```bash
git clone -b gpu-mining https://github.com/OSXBasedAnon/alphanumeric.git
cd alphanumeric
cargo build --release --features gpu_miner
# then: mine <wallet> --gpu
```

That path is **solo against your local node**, not pool stratum. This repo is for **pool mining**.

## multipooldd.com note

[multipooldd.com](https://multipooldd.com/dashboard) is a multipool dashboard (currently JUNO/QRL/XMR RandomX). It is linked as an **example of pool UX**, not as an ALPHA endpoint. ALPHA stratum today is **rplant :7176** (`blake3-an`).

## License

MIT. BLAKE3 C code is Apache-2.0 / CC0 (see `third_party/blake3`).
