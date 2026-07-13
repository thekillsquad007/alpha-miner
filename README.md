# alpha-miner (modular multi-coin)

Open-source **multi-coin GPU/CPU pool miner** with a plugin-style coin registry.

| Coin | Algo | Protocol | Backends |
|------|------|----------|----------|
| **Lattica (LTA)** | `sha3d` — SHA3-256d over **80-byte** header | Bitcoin stratum v1 | **CUDA**, OpenCL, CPU |
| **Alphanumeric (ALPHA)** | `blake3-an` — BLAKE3 over **92-byte** header | Monero-style stratum | CUDA, **HIP**, OpenCL, CPU |

Consensus references:

- Lattica: [lattica-core/lattica](https://github.com/lattica-core/lattica) — `CBlockHeader::GetPoWHash` = SHA3-256d (FIPS 202)
- ALPHA: [OSXBasedAnon/alphanumeric](https://github.com/OSXBasedAnon/alphanumeric) `gpu-mining`

## Architecture (add a coin without a rewrite)

```
src/
  core_types.hpp      Job / Share / IWorkSource / IShareSink / IMinerBackend
  coin_registry.cpp   CoinProfile table (id, algo, protocol, header layout)
  sha3.cpp            Portable SHA3-256 + SHA3-256d (Lattica PoW)
  stratum.cpp         XMR-style login/job/submit (ALPHA)
  stratum_btc.cpp     Bitcoin mining.subscribe/notify/submit (Lattica pools)
  backends            CPU / OpenCL / CUDA / HIP dispatch by AlgoId
kernels/
  blake3_an.cl
  sha3d.cl            Unrolled Keccak-f[1600] (matches Lattica in-tree miner)
```

**To add a new coin that reuses an existing algo:** append a `CoinProfile` in `coin_registry.cpp`.

**To add a new algo:** implement host hash + target compare, add OpenCL/CUDA kernel, and branch in the backend workers.

## Quick start — Lattica (SHA3-256d)

```bash
# Build (CUDA preferred on NVIDIA; OpenCL for AMD/Intel; CPU always)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Self-test
./build/alpha-miner --list-coins
./build/alpha-miner --benchmark -c lattica

# Pool mine (Bitcoin stratum — e.g. coin-miners.info / LedgeRock)
./build/alpha-miner -c lattica -o HOST:PORT -u lta1qYOURADDRESS.rig1 -b cuda
./build/alpha-miner -c lattica -o HOST:PORT -u solo:lta1qYOURADDRESS -b opencl -k kernels

# Multi-GPU
./build/alpha-miner -c lattica -o HOST:PORT -u lta1q....rig1 -b cuda -d 0,1,2
```

Lattica PoW is **SHA3-256(SHA3-256(header[80]))** with LE uint256 target compare — bit-identical to the node’s `GetPoWHash`. CUDA and OpenCL kernels use a fully unrolled Keccak-f[1600] (same approach as the official `lattica-miner` OpenCL path) for maximum hashrate.

## Quick start — ALPHA (blake3-an)

```bash
# NVIDIA CUDA
./build/alpha-miner -c alpha -o eu.rplant.xyz:7176 -u YOUR_ALPHA.rig1 -b cuda

# AMD HIP (ROCm)
source scripts/env-wsl-hip.sh   # WSL only
bash scripts/build-hip.sh
./build-hip/alpha-miner -c alpha -o eu.rplant.xyz:7176 -u YOUR_ALPHA.rig1 -b hip
```

ALPHA developer fee: **2%** (second stratum session). Lattica has **no built-in devfee**.

## Backends

| Backend | Hardware | Build flag | Algos |
|---------|----------|------------|-------|
| **CUDA** | NVIDIA multi-GPU (sm_75–sm_120) | `-DALPHA_MINER_CUDA=ON` | blake3-an, **sha3d** |
| **HIP** | AMD Radeon (ROCm) | `-DALPHA_MINER_HIP=ON` | blake3-an |
| **OpenCL** | AMD + NVIDIA + Intel | `-DALPHA_MINER_OPENCL=ON` | blake3-an, **sha3d** |
| **CPU** | any | always | blake3-an, sha3d |

Auto backend order: CUDA → HIP (blake3 only) → OpenCL → CPU.

## CLI

```
-c, --coin NAME         alpha | lattica
-a, --algo NAME         blake3-an | sha3d
-o, --url HOST:PORT     stratum endpoint
-u, --user USER         wallet[.worker]
-p, --pass PASS         password (default x)
-b, --backend NAME      cpu | cuda | hip | opencl | auto
-d, --devices LIST      0,1,2
-k, --kernel-dir PATH   directory with blake3_an.cl / sha3d.cl
-l, --list-devices
    --list-coins
    --benchmark
```

## Header layouts

### Lattica (80 bytes)

```
offset  size  field
0       4     nVersion (u32 LE)
4       32    hashPrevBlock
36      32    hashMerkleRoot
68      4     nTime (u32 LE)
72      4     nBits (u32 LE)
76      4     nNonce (u32 LE)   ← miner searches full 32-bit space
= 80 bytes → SHA3-256d → LE uint256 ≤ target
```

Stratum submit: `mining.submit [worker, job_id, extranonce2, ntime, nonce]`.

### ALPHA (92 bytes)

```
offset  size  field
0       4     index
4       32    previous_hash
36      8     timestamp
44      8     nonce (u64 LE)   ← low 48 free; high 16 pool extranonce
52      8     difficulty
60      32    merkle_root
= 92 bytes → BLAKE3 → BE byte-compare ≤ target
```

## License

MIT. BLAKE3 C code is Apache-2.0 / CC0 (see `third_party/blake3`).
