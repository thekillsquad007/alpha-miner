# alpha-miner — usage (multi-coin)

Modular miner: **Lattica (sha3d)** and **Alphanumeric (blake3-an)**.

| Flag | Meaning |
|------|---------|
| `-c COIN` | `lattica` (default) \| `alpha` |
| `-a ALGO` | `sha3d` \| `blake3-an` (optional override) |
| `-o HOST:PORT` | Pool stratum |
| `-u WALLET.worker` | Stratum user (`solo:addr` on some LTA pools) |
| `-p PASS` | Password (default `x`) |
| `-b BACKEND` | `auto` \| `cuda` \| `hip` \| `opencl` \| `cpu` |
| `-d 0,1,2` | GPU indices |
| `-k DIR` | Kernel directory (`blake3_an.cl`, `sha3d.cl`) |
| `-l` | List GPUs |
| `--list-coins` | List coin plugins |
| `--benchmark` | CPU algo self-test / hashrate |

`auto` order: **CUDA → HIP (blake3 only) → OpenCL → CPU**.

---

## Lattica (LTA) — SHA3-256d

**Algo:** double SHA3-256 over 80-byte Bitcoin-style header (matches [lattica-core/lattica](https://github.com/lattica-core/lattica)).  
**Protocol:** Bitcoin stratum v1.  
**Address:** bech32 `lta1q…`  
**Fee in this miner:** none.

```bash
# NVIDIA CUDA (highest hashrate on GeForce)
./alpha-miner -c lattica -o POOL_HOST:PORT -u lta1qYOURADDRESS.rig1 -b cuda

# Multi-GPU
./alpha-miner -c lattica -o POOL_HOST:PORT -u lta1qYOURADDRESS.rig1 -b cuda -d 0,1,2

# OpenCL (AMD / Intel / NVIDIA fallback)
./alpha-miner -c lattica -o POOL_HOST:PORT -u lta1qYOURADDRESS.rig1 -b opencl -k kernels

# Pool solo mode (coin-miners.info style)
./alpha-miner -c lattica -o POOL_HOST:PORT -u solo:lta1qYOURADDRESS -b cuda

# CPU only
./alpha-miner -c lattica -o POOL_HOST:PORT -u lta1qYOURADDRESS -b cpu -t 16
```

Public pools (check their “How to mine” page for current stratum hosts/ports):

- [lattica.coin-miners.info](https://lattica.coin-miners.info/) — PPLNS / solo, 1%
- LedgeRock multi-coin pool — see BitcoinTalk Lattica ANN

Build with CUDA:

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DALPHA_MINER_CUDA=ON
cmake --build build-cuda -j$(nproc)
```

---

## Alphanumeric (ALPHA) — blake3-an

**Pool:** `stratum+tcp://eu.rplant.xyz:7176` (also `na` / `asia`)  
**Fee:** 2% built-in second stratum session  
**Wallet:** 40-hex ALPHA address

```bash
# NVIDIA
./alpha-miner -c alpha -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b cuda

# AMD HIP
bash scripts/build-hip.sh
./build-hip/alpha-miner -c alpha -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b hip

# OpenCL / CPU
./alpha-miner -c alpha -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b opencl -k kernels
./alpha-miner -c alpha -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b cpu -t 16
```

---

## Adding another coin

1. If the **algo already exists** (`sha3d` / `blake3-an`): add a `CoinProfile` in `src/coin_registry.cpp` (id, protocol, header layout, target mode).
2. If the **algo is new**: host hash in `src/`, OpenCL kernel under `kernels/`, CUDA path in `cuda_miner.cu`, then register the coin.
3. Protocol: reuse `stratum_btc.cpp` (Bitcoin) or `stratum.cpp` (Monero-style), or add a new `IWorkSource`.

No need to rewrite the mining loop or job mux.
