# alpha-miner — usage (all platforms)

**Algorithm:** `blake3-an` · **Fee:** 2% (built-in second stratum session)  
**Pool:** `stratum+tcp://eu.rplant.xyz:7176` (also `na.rplant.xyz` / `asia.rplant.xyz`)  
**Wallet:** 40-hex ALPHA address, optional `.worker` suffix

Common options:

| Flag | Meaning |
|------|---------|
| `-o HOST:PORT` | Pool (with or without `stratum+tcp://`) |
| `-u WALLET.worker` | Stratum user |
| `-p PASS` | Password (default `x`) |
| `-b BACKEND` | `auto` \| `cuda` \| `hip` \| `opencl` \| `cpu` |
| `-d 0,1,2` | GPU indices (multi-GPU) |
| `-k PATH` | OpenCL kernel (`blake3_an.cl`) |
| `-t N` | CPU threads |
| `-l` | List detected GPUs and exit |
| `-h` | Help |

`auto` order: **CUDA → HIP → OpenCL → CPU**.

---

## 1. NVIDIA (Linux) — CUDA multi-GPU

**Asset:** `alpha-miner-linux-cuda-x64-v*.tar.gz`  
**GPUs:** RTX 20/30/40/50-series (sm_75–sm_120), e.g. 3060/3070/3080/4070/4080/**5080**

```bash
tar xzf alpha-miner-linux-cuda-x64-v0.2.4.tar.gz
chmod +x alpha-miner-linux-cuda-x64

# List GPUs
./alpha-miner-linux-cuda-x64 -l

# All GPUs (recommended)
./alpha-miner-linux-cuda-x64 \
  -o eu.rplant.xyz:7176 \
  -u YOUR_ALPHA_ADDRESS.rig1 \
  -p x \
  -b cuda

# Select devices (example: first three)
./alpha-miner-linux-cuda-x64 \
  -o eu.rplant.xyz:7176 \
  -u YOUR_ALPHA_ADDRESS.rig1 \
  -b cuda -d 0,1,2

# Asia / NA pools
./alpha-miner-linux-cuda-x64 -o asia.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b cuda
./alpha-miner-linux-cuda-x64 -o na.rplant.xyz:7176   -u YOUR_ALPHA_ADDRESS.rig1 -b cuda
```

**Needs:** NVIDIA driver + CUDA runtime (driver 550+ recommended; 50-series needs recent driver).

Build from source:

```bash
export PATH=/usr/local/cuda/bin:$PATH
bash scripts/build-cuda.sh
./build-cuda/alpha-miner -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b cuda
```

---

## 2. AMD (Linux) — HIP / ROCm

**Best path for Radeon** (RX 6000/7000, WSL2 with DXG).  
**Asset:** build from source (or use OpenCL package below as fallback).

```bash
# Native Linux ROCm
bash scripts/build-hip.sh
./build-hip/alpha-miner -l
./build-hip/alpha-miner -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b hip

# Multi-GPU
./build-hip/alpha-miner -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b hip -d 0,1
```

**WSL2 (AMD):**

```bash
source scripts/env-wsl-hip.sh   # HSA_ENABLE_DXG_DETECTION + librocdxg
bash scripts/build-hip.sh
./build-hip/alpha-miner -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b hip
```

---

## 3. AMD or NVIDIA (Linux) — OpenCL

**Asset:** `alpha-miner-ubuntu-22.04-x64.tar.gz`  
Works without CUDA/ROCm toolkits (uses vendor OpenCL ICD).

```bash
tar xzf alpha-miner-ubuntu-22.04-x64.tar.gz
chmod +x alpha-miner-ubuntu-22.04-x64

# Keep blake3_an.cl next to the binary (or pass -k)
./alpha-miner-ubuntu-22.04-x64 -l

./alpha-miner-ubuntu-22.04-x64 \
  -o eu.rplant.xyz:7176 \
  -u YOUR_ALPHA_ADDRESS.rig1 \
  -p x \
  -b opencl \
  -k blake3_an.cl

# Multi-GPU
./alpha-miner-ubuntu-22.04-x64 \
  -o eu.rplant.xyz:7176 \
  -u YOUR_ALPHA_ADDRESS.rig1 \
  -b opencl -d 0,1,2 -k blake3_an.cl
```

---

## 4. Windows (x64)

**Asset:** `alpha-miner-windows-x64.zip` (OpenCL build)

```powershell
Expand-Archive alpha-miner-windows-x64.zip -DestinationPath alpha-miner
cd alpha-miner

.\alpha-miner-windows-x64.exe -l

.\alpha-miner-windows-x64.exe `
  -o eu.rplant.xyz:7176 `
  -u YOUR_ALPHA_ADDRESS.rig1 `
  -p x `
  -b opencl `
  -k blake3_an.cl

# Multi-GPU
.\alpha-miner-windows-x64.exe -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b opencl -d 0,1
```

**NVIDIA on Windows:** install [GPU drivers](https://www.nvidia.com/drivers) with OpenCL, or build CUDA locally with VS + CUDA Toolkit:

```bat
cmake -S . -B build-cuda -DALPHA_MINER_CUDA=ON -DALPHA_MINER_HIP=OFF -DALPHA_MINER_OPENCL=OFF
cmake --build build-cuda --config Release
build-cuda\Release\alpha-miner.exe -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b cuda
```

**AMD on Windows:** Adrenalin drivers + OpenCL path above.

---

## 5. HiveOS (custom miner)

**Asset:** `alpha-hiveos-*.tar.gz` (folder name must be `alpha`)

1. HiveOS → **Workers** → **GPU** → install custom miner from URL, or upload the tarball.  
2. Flight sheet:

| Field | Value |
|-------|--------|
| **Coin** | (any / custom) |
| **Wallet / Template** | `YOUR_ALPHA_ADDRESS.worker` |
| **Pool URL** | `stratum+tcp://eu.rplant.xyz:7176` |
| **Password** | `x` |
| **Extra config** | see below |

**NVIDIA (prefer CUDA if you install the CUDA binary into the package, else OpenCL):**

```text
-b opencl
```

or multi-GPU / force devices:

```text
-b opencl -d 0,1,2,3
```

**AMD:**

```text
-b opencl
```

Leave Extra empty for `auto` (CUDA if linked, else OpenCL/CPU).

**HiveOS shell (manual run after install):**

```bash
cd /hive/miners/custom/alpha
./alpha -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b opencl -k blake3_an.cl
```

---

## 6. CPU only (any OS)

```bash
# Linux (ubuntu package or any build)
./alpha-miner-ubuntu-22.04-x64 -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.cpu1 -b cpu -t 16

# Windows
.\alpha-miner-windows-x64.exe -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.cpu1 -b cpu -t 8
```

---

## 7. Which asset do I download?

| Hardware | OS | Download |
|----------|-----|----------|
| NVIDIA GPU(s) | Linux | **`alpha-miner-linux-cuda-x64-*.tar.gz`** |
| AMD GPU(s) | Linux | Build HIP **or** `alpha-miner-ubuntu-22.04-x64.tar.gz` (OpenCL) |
| NVIDIA / AMD | Windows | `alpha-miner-windows-x64.zip` (OpenCL) |
| HiveOS farm | HiveOS | `alpha-hiveos-*.tar.gz` |
| No GPU | Linux/Windows | Ubuntu or Windows binary with `-b cpu` |

---

## 8. Tips

- Username = **payout address** (40 hex). Optional `.worker` for dashboard.
- Multi-GPU: always use `-d 0,1,...` if you want a subset; omit `-d` to use **all** devices.
- List devices first: `-l`.
- Keep `blake3_an.cl` beside OpenCL binaries (or pass `-k /full/path/blake3_an.cl`).
- Firewall: outbound TCP **7176** to the pool.
- 2% devfee runs a second stratum session; hashrate stats include fee work.

## Support pools

```text
stratum+tcp://eu.rplant.xyz:7176
stratum+tcp://na.rplant.xyz:7176
stratum+tcp://asia.rplant.xyz:7176
```
