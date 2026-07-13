// HIP GPU miner for Alphanumeric blake3-an (AMD ROCm).
// Compiles with: hipcc -O3 -c hip_miner.hip
//
// Header: 92 bytes LE (index|prev|ts|nonce|diff|merkle) → BLAKE3 → 32-byte hash.
// Nonce: low 48 bits free to search; high 16 bits are pool extranonce.

#include "hip_miner.hpp"
#include "util.hpp"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

namespace alpha {
namespace {

#define HIP_CHECK(call)                                                          \
  do {                                                                           \
    hipError_t _e = (call);                                                      \
    if (_e != hipSuccess) {                                                      \
      throw std::runtime_error(std::string("HIP: ") + hipGetErrorString(_e) +    \
                               " @ " + #call);                                   \
    }                                                                            \
  } while (0)

// ---- Device BLAKE3 (92-byte one-shot) ----

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32u - n));
}

__device__ __forceinline__ void g_qr(uint32_t* s, int a, int b, int c, int d,
                                    uint32_t mx, uint32_t my) {
  s[a] = s[a] + s[b] + mx;
  s[d] = rotr32(s[d] ^ s[a], 16u);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 12u);
  s[a] = s[a] + s[b] + my;
  s[d] = rotr32(s[d] ^ s[a], 8u);
  s[c] = s[c] + s[d];
  s[b] = rotr32(s[b] ^ s[c], 7u);
}

__device__ void blake3_compress(const uint32_t cv[8], const uint32_t block[16],
                                uint32_t block_len, uint32_t flags,
                                uint32_t out[8]) {
  constexpr uint32_t IV0 = 0x6A09E667u, IV1 = 0xBB67AE85u, IV2 = 0x3C6EF372u,
                     IV3 = 0xA54FF53Au, IV4 = 0x510E527Fu, IV5 = 0x9B05688Cu,
                     IV6 = 0x1F83D9ABu, IV7 = 0x5BE0CD19u;
  uint32_t s[16];
  s[0] = cv[0];
  s[1] = cv[1];
  s[2] = cv[2];
  s[3] = cv[3];
  s[4] = cv[4];
  s[5] = cv[5];
  s[6] = cv[6];
  s[7] = cv[7];
  s[8] = IV0;
  s[9] = IV1;
  s[10] = IV2;
  s[11] = IV3;
  s[12] = 0u;
  s[13] = 0u;
  s[14] = block_len;
  s[15] = flags;

  uint32_t m0 = block[0], m1 = block[1], m2 = block[2], m3 = block[3];
  uint32_t m4 = block[4], m5 = block[5], m6 = block[6], m7 = block[7];
  uint32_t m8 = block[8], m9 = block[9], m10 = block[10], m11 = block[11];
  uint32_t m12 = block[12], m13 = block[13], m14 = block[14], m15 = block[15];

  for (int round = 0; round < 7; ++round) {
    g_qr(s, 0, 4, 8, 12, m0, m1);
    g_qr(s, 1, 5, 9, 13, m2, m3);
    g_qr(s, 2, 6, 10, 14, m4, m5);
    g_qr(s, 3, 7, 11, 15, m6, m7);
    g_qr(s, 0, 5, 10, 15, m8, m9);
    g_qr(s, 1, 6, 11, 12, m10, m11);
    g_qr(s, 2, 7, 8, 13, m12, m13);
    g_qr(s, 3, 4, 9, 14, m14, m15);
    if (round < 6) {
      uint32_t n0 = m2, n1 = m6, n2 = m3, n3 = m10, n4 = m7, n5 = m0, n6 = m4, n7 = m13;
      uint32_t n8 = m1, n9 = m11, n10 = m12, n11 = m5, n12 = m9, n13 = m14, n14 = m15,
               n15 = m8;
      m0 = n0;
      m1 = n1;
      m2 = n2;
      m3 = n3;
      m4 = n4;
      m5 = n5;
      m6 = n6;
      m7 = n7;
      m8 = n8;
      m9 = n9;
      m10 = n10;
      m11 = n11;
      m12 = n12;
      m13 = n13;
      m14 = n14;
      m15 = n15;
    }
  }
  out[0] = s[0] ^ s[8];
  out[1] = s[1] ^ s[9];
  out[2] = s[2] ^ s[10];
  out[3] = s[3] ^ s[11];
  out[4] = s[4] ^ s[12];
  out[5] = s[5] ^ s[13];
  out[6] = s[6] ^ s[14];
  out[7] = s[7] ^ s[15];
}

__device__ void blake3_92(const uint32_t w[23], uint32_t nonce_lo, uint32_t nonce_hi,
                         uint32_t out[8]) {
  constexpr uint32_t IV0 = 0x6A09E667u, IV1 = 0xBB67AE85u, IV2 = 0x3C6EF372u,
                     IV3 = 0xA54FF53Au, IV4 = 0x510E527Fu, IV5 = 0x9B05688Cu,
                     IV6 = 0x1F83D9ABu, IV7 = 0x5BE0CD19u;
  constexpr uint32_t CHUNK_START = 1u, CHUNK_END = 2u, ROOT = 8u;

  uint32_t cv[8] = {IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7};
  uint32_t block0[16];
  block0[0] = w[0];
  block0[1] = w[1];
  block0[2] = w[2];
  block0[3] = w[3];
  block0[4] = w[4];
  block0[5] = w[5];
  block0[6] = w[6];
  block0[7] = w[7];
  block0[8] = w[8];
  block0[9] = w[9];
  block0[10] = w[10];
  block0[11] = nonce_lo;
  block0[12] = nonce_hi;
  block0[13] = w[13];
  block0[14] = w[14];
  block0[15] = w[15];
  uint32_t mid[8];
  blake3_compress(cv, block0, 64u, CHUNK_START, mid);

  uint32_t block1[16] = {};
  block1[0] = w[16];
  block1[1] = w[17];
  block1[2] = w[18];
  block1[3] = w[19];
  block1[4] = w[20];
  block1[5] = w[21];
  block1[6] = w[22];
  blake3_compress(mid, block1, 28u, CHUNK_END | ROOT, out);
}

// Compare hash <= target as big-endian byte strings.
// Words store little-endian bytes of the sequential hash/target streams.
__device__ __forceinline__ bool meets_target(const uint32_t h[8],
                                             const uint32_t target[8]) {
  for (int i = 0; i < 8; ++i) {
    uint32_t hw = h[i];
    uint32_t tw = target[i];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      uint8_t hb = (uint8_t)((hw >> (8 * b)) & 0xff);
      uint8_t tb = (uint8_t)((tw >> (8 * b)) & 0xff);
      if (hb < tb) return true;
      if (hb > tb) return false;
    }
  }
  return true;
}

// result[0]=found flag, [1]=nonce_lo, [2]=nonce_hi, [3..10]=hash words
__global__ void search_kernel(const uint32_t* __restrict__ header_words,
                              const uint32_t* __restrict__ target_words,
                              uint32_t base_lo, uint32_t base_hi, uint32_t iters,
                              uint32_t* __restrict__ result) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t gsz = gridDim.x * blockDim.x;
  const uint64_t base = (static_cast<uint64_t>(base_hi) << 32) | base_lo;
  const uint64_t start = base + gid;

  uint32_t w[23];
#pragma unroll
  for (int i = 0; i < 23; ++i) w[i] = header_words[i];

  uint32_t t[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) t[i] = target_words[i];

  for (uint32_t i = 0; i < iters; ++i) {
    if (atomicAdd(&result[0], 0) != 0) return;
    const uint64_t nonce = start + static_cast<uint64_t>(i) * gsz;
    const uint32_t nlo = static_cast<uint32_t>(nonce);
    const uint32_t nhi = static_cast<uint32_t>(nonce >> 32);
    uint32_t h[8];
    blake3_92(w, nlo, nhi, h);
    if (meets_target(h, t)) {
      if (atomicCAS(&result[0], 0u, 1u) == 0u) {
        result[1] = nlo;
        result[2] = nhi;
#pragma unroll
        for (int k = 0; k < 8; ++k) result[3 + k] = h[k];
      }
      return;
    }
  }
}

// GPU self-check: hash one fixed header, write 8 words.
__global__ void hash_one_kernel(const uint32_t* __restrict__ header_words,
                                uint32_t nonce_lo, uint32_t nonce_hi,
                                uint32_t* __restrict__ out_hash) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  uint32_t w[23];
  for (int i = 0; i < 23; ++i) w[i] = header_words[i];
  uint32_t h[8];
  blake3_92(w, nonce_lo, nonce_hi, h);
  for (int i = 0; i < 8; ++i) out_hash[i] = h[i];
}

void words_from_bytes(const uint8_t* bytes, int nbytes, uint32_t* words) {
  const int nwords = (nbytes + 3) / 4;
  for (int i = 0; i < nwords; ++i) {
    uint32_t w = 0;
    for (int b = 0; b < 4; ++b) {
      int idx = i * 4 + b;
      if (idx < nbytes) w |= static_cast<uint32_t>(bytes[idx]) << (8 * b);
    }
    words[i] = w;
  }
}

}  // namespace

// ---- HipMiner ----

// Detect usable HIP stack:
//  - native Linux: /dev/kfd (+ optional /dev/dri)
//  - WSL2 AMD:     /dev/dxg via librocdxg (HSA_ENABLE_DXG_DETECTION=1)
// Never assume "no GPU" solely because /dev/kfd is missing.
static bool hip_transport_present() {
  if (access("/dev/kfd", F_OK) == 0) return true;
  if (access("/dev/dxg", F_OK) == 0) return true;  // WSL DXG bridge
  if (access("/dev/dri/renderD128", F_OK) == 0) return true;
  return false;
}

int HipMiner::device_count() {
  // Always try the runtime — hipGetDeviceCount is the source of truth.
  // Transport check is only a fast-path skip when nothing is mounted.
  if (!hip_transport_present()) {
    // Still attempt once; some containers expose GPU without those nodes.
  }
  int n = 0;
  hipError_t e = hipGetDeviceCount(&n);
  if (e != hipSuccess || n < 0) return 0;
  return n;
}

std::vector<std::string> HipMiner::list_devices() {
  std::vector<std::string> out;
  int n = 0;
  hipError_t e = hipGetDeviceCount(&n);
  if (e != hipSuccess) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "(hipGetDeviceCount failed: %s; transport kfd=%d dxg=%d)",
                  hipGetErrorString(e),
                  access("/dev/kfd", F_OK) == 0,
                  access("/dev/dxg", F_OK) == 0);
    out.emplace_back(buf);
    return out;
  }
  if (n <= 0) {
    out.emplace_back("(0 HIP devices — set LD_LIBRARY_PATH to include librocdxg "
                     "on WSL, e.g. /opt/rocm-7.2.3/lib, and HSA_ENABLE_DXG_DETECTION=1)");
    return out;
  }
  for (int i = 0; i < n; ++i) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, i) != hipSuccess) continue;
    char buf[384];
    std::snprintf(buf, sizeof(buf), "%d: %s (cu=%d, %.0f MiB, %s)", i, prop.name,
                  prop.multiProcessorCount, prop.totalGlobalMem / (1024.0 * 1024.0),
                  prop.gcnArchName);
    out.emplace_back(buf);
  }
  return out;
}

HipMiner::HipMiner(miner::JobMux& jobs, miner::IShareSink& sink, const std::vector<int>& devices,
                   miner::AlgoId algo)
    : jobs_(jobs), sink_(sink), devices_(devices), algo_(algo) {}

HipMiner::~HipMiner() { stop(); }

bool HipMiner::init() {
  if (algo_ != miner::AlgoId::Blake3An) {
    std::cerr << "[hip] only blake3-an on HIP; use OpenCL/CUDA for sha3d (Lattica)\n";
    return false;
  }
  int n = device_count();
  if (n <= 0) {
    std::cerr << "[hip] no AMD GPU devices (hipGetDeviceCount=" << n << ")\n";
    return false;
  }
  std::cout << "[hip] devices:\n";
  for (auto& s : list_devices()) std::cout << "  " << s << "\n";

  // Self-check on device 0: GPU BLAKE3 must match CPU.
  try {
    HIP_CHECK(hipSetDevice(0));
    uint8_t hdr[92];
    for (int i = 0; i < 92; ++i) hdr[i] = static_cast<uint8_t>(i * 37 + 11);
    uint64_t nonce = 0x0123456789ABCDEFULL;
    for (int i = 0; i < 8; ++i) hdr[44 + i] = static_cast<uint8_t>((nonce >> (8 * i)) & 0xff);

    uint32_t words[23] = {};
    words_from_bytes(hdr, 92, words);
    // zero nonce slots — kernel uses args
    words[11] = 0;
    words[12] = 0;

    uint32_t *d_w = nullptr, *d_h = nullptr;
    HIP_CHECK(hipMalloc(&d_w, sizeof(words)));
    HIP_CHECK(hipMalloc(&d_h, 8 * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(d_w, words, sizeof(words), hipMemcpyHostToDevice));
    hash_one_kernel<<<1, 1>>>(d_w, static_cast<uint32_t>(nonce),
                              static_cast<uint32_t>(nonce >> 32), d_h);
    HIP_CHECK(hipDeviceSynchronize());
    uint32_t gpu_words[8];
    HIP_CHECK(hipMemcpy(gpu_words, d_h, sizeof(gpu_words), hipMemcpyDeviceToHost));
    (void)hipFree(d_w);
    (void)hipFree(d_h);

    uint8_t gpu_hash[32];
    for (int i = 0; i < 8; ++i) {
      gpu_hash[i * 4 + 0] = static_cast<uint8_t>(gpu_words[i] & 0xff);
      gpu_hash[i * 4 + 1] = static_cast<uint8_t>((gpu_words[i] >> 8) & 0xff);
      gpu_hash[i * 4 + 2] = static_cast<uint8_t>((gpu_words[i] >> 16) & 0xff);
      gpu_hash[i * 4 + 3] = static_cast<uint8_t>((gpu_words[i] >> 24) & 0xff);
    }
    uint8_t cpu_hash[32];
    blake3_header(hdr, cpu_hash);
    if (std::memcmp(gpu_hash, cpu_hash, 32) != 0) {
      std::cerr << "[hip] self-check FAILED\n"
                << "  cpu=" << to_hex(cpu_hash, 32) << "\n"
                << "  gpu=" << to_hex(gpu_hash, 32) << "\n";
      return false;
    }
    std::cout << "[hip] self-check OK (" << to_hex(cpu_hash, 8) << "...)\n";
  } catch (const std::exception& e) {
    std::cerr << "[hip] init/self-check error: " << e.what() << "\n";
    return false;
  }

  ready_ = true;
  return true;
}

void HipMiner::start() {
  if (!ready_) return;
  stop_ = false;
  hashes_ = 0;
  start_ms_ = now_ms();
  int n = device_count();
  std::vector<int> use = devices_;
  if (use.empty()) {
    for (int i = 0; i < n; ++i) use.push_back(i);
  }
  for (size_t i = 0; i < use.size(); ++i) {
    workers_.emplace_back([this, dev = use[i], i] { worker(dev, static_cast<int>(i)); });
  }
}

void HipMiner::stop() {
  stop_ = true;
  for (auto& t : workers_)
    if (t.joinable()) t.join();
  workers_.clear();
}

double HipMiner::hashrate() const {
  uint64_t ms = now_ms() - start_ms_;
  if (!ms) return 0;
  return static_cast<double>(hashes_.load()) * 1000.0 / static_cast<double>(ms);
}

void HipMiner::worker(int device_id, int logical_id) {
  try {
    HIP_CHECK(hipSetDevice(device_id));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    std::cout << "[hip] mining on device " << device_id << " (" << prop.name << ")\n";

    // ~256k threads, 64 iters → ~16M hashes/dispatch (tune per card)
    const int BLOCK = 256;
    const int GRID = 1024;  // 1024*256 = 262144
    const uint32_t ITERS = 64;
    const uint64_t PER_DISPATCH =
        static_cast<uint64_t>(GRID) * BLOCK * ITERS;

    uint32_t *d_header = nullptr, *d_target = nullptr, *d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_header, 23 * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_target, 8 * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_result, 11 * sizeof(uint32_t)));

    while (!stop_) {
      Job job = jobs_.get();
      if (job.job_id.empty() || job.epoch == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      const uint64_t epoch = job.epoch;
      const bool fee = job.is_devfee;
      const uint64_t hi = job.nonce_fixed;

      uint32_t words[23] = {};
      words_from_bytes(job.header.data(), 92, words);
      words[11] = 0;
      words[12] = 0;
      uint32_t twords[8] = {};
      words_from_bytes(job.target.data(), 32, twords);

      HIP_CHECK(hipMemcpy(d_header, words, sizeof(words), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_target, twords, sizeof(twords), hipMemcpyHostToDevice));

      uint64_t cursor = hi | (static_cast<uint64_t>(logical_id) << 40);

      while (!stop_) {
        Job latest = jobs_.get();
        if (latest.job_id != job.job_id || latest.epoch != epoch || latest.is_devfee != fee) break;

        uint32_t zero[11] = {};
        HIP_CHECK(hipMemcpy(d_result, zero, sizeof(zero), hipMemcpyHostToDevice));

        uint32_t base_lo = static_cast<uint32_t>(cursor);
        uint32_t base_hi = static_cast<uint32_t>(cursor >> 32);
        search_kernel<<<GRID, BLOCK>>>(d_header, d_target, base_lo, base_hi, ITERS, d_result);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        uint32_t res[11] = {};
        HIP_CHECK(hipMemcpy(res, d_result, sizeof(res), hipMemcpyDeviceToHost));
        hashes_.fetch_add(PER_DISPATCH, std::memory_order_relaxed);

        if (res[0]) {
          uint64_t nonce = (static_cast<uint64_t>(res[2]) << 32) | res[1];
          // Preserve extranonce binding
          if ((nonce & 0xFFFF000000000000ULL) != hi) {
            // shouldn't happen if cursor kept hi bits
          }
          uint8_t header[92];
          std::memcpy(header, job.header.data(), 92);
          for (int i = 0; i < 8; ++i)
            header[44 + i] = static_cast<uint8_t>((nonce >> (8 * i)) & 0xff);
          uint8_t hash[32];
          blake3_header(header, hash);
          if (hash_meets_job_target(hash, job)) {
            Share s;
            s.job_id = job.job_id;
            s.nonce = nonce;
            std::memcpy(s.hash.data(), hash, 32);
            s.job_epoch = epoch;
            s.is_devfee = fee;
            std::cout << (fee ? "[hip/fee] " : "[hip] ") << "share nonce=" << to_hex(header + 44, 8)
                      << " hash=" << to_hex(hash, 8) << "...\n";
            sink_.submit(s);
          } else {
            std::cerr << "[hip] GPU hit failed CPU verify (ignored)\n";
          }
        }

        cursor += PER_DISPATCH;
        cursor = (cursor & 0x0000FFFFFFFFFFFFULL) | hi;
      }
    }

    (void)hipFree(d_header);
    (void)hipFree(d_target);
    (void)hipFree(d_result);
  } catch (const std::exception& e) {
    std::cerr << "[hip] worker " << device_id << " error: " << e.what() << "\n";
  }
}

}  // namespace alpha
