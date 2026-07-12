// CUDA multi-GPU miner for Alphanumeric blake3-an (NVIDIA).
// Same 92-byte header BLAKE3 PoW as HIP/OpenCL backends.
// One worker thread per selected CUDA device.

#include "cuda_miner.hpp"
#include "util.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace alpha {
namespace {

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t _e = (call);                                                     \
    if (_e != cudaSuccess) {                                                     \
      throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(_e) +  \
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

// result[0]=found, [1]=nonce_lo, [2]=nonce_hi, [3..10]=hash words
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

// ---- CudaMiner ----

int CudaMiner::device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n < 0 ? 0 : n;
}

std::vector<std::string> CudaMiner::list_devices() {
  std::vector<std::string> out;
  int n = device_count();
  if (n <= 0) {
    out.emplace_back("(no CUDA devices)");
    return out;
  }
  for (int i = 0; i < n; ++i) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
    char buf[384];
    std::snprintf(buf, sizeof(buf), "%d: %s (sm_%d%d, %d SMs, %.0f MiB)", i, prop.name,
                  prop.major, prop.minor, prop.multiProcessorCount,
                  prop.totalGlobalMem / (1024.0 * 1024.0));
    out.emplace_back(buf);
  }
  return out;
}

CudaMiner::CudaMiner(JobMux& jobs, ShareRouter& router, const std::vector<int>& devices)
    : jobs_(jobs), router_(router), devices_(devices) {}

CudaMiner::~CudaMiner() { stop(); }

bool CudaMiner::init() {
  int n = device_count();
  if (n <= 0) {
    std::cerr << "[cuda] no NVIDIA devices (cudaGetDeviceCount=" << n << ")\n";
    return false;
  }
  std::cout << "[cuda] devices:\n";
  for (auto& s : list_devices()) std::cout << "  " << s << "\n";

  try {
    CUDA_CHECK(cudaSetDevice(0));
    uint8_t hdr[92];
    for (int i = 0; i < 92; ++i) hdr[i] = static_cast<uint8_t>(i * 37 + 11);
    uint64_t nonce = 0x0123456789ABCDEFULL;
    for (int i = 0; i < 8; ++i) hdr[44 + i] = static_cast<uint8_t>((nonce >> (8 * i)) & 0xff);

    uint32_t words[23] = {};
    words_from_bytes(hdr, 92, words);
    words[11] = 0;
    words[12] = 0;

    uint32_t *d_w = nullptr, *d_h = nullptr;
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(words)));
    CUDA_CHECK(cudaMalloc(&d_h, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_w, words, sizeof(words), cudaMemcpyHostToDevice));
    hash_one_kernel<<<1, 1>>>(d_w, static_cast<uint32_t>(nonce),
                              static_cast<uint32_t>(nonce >> 32), d_h);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t gpu_words[8];
    CUDA_CHECK(cudaMemcpy(gpu_words, d_h, sizeof(gpu_words), cudaMemcpyDeviceToHost));
    cudaFree(d_w);
    cudaFree(d_h);

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
      std::cerr << "[cuda] self-check FAILED\n"
                << "  cpu=" << to_hex(cpu_hash, 32) << "\n"
                << "  gpu=" << to_hex(gpu_hash, 32) << "\n";
      return false;
    }
    std::cout << "[cuda] self-check OK (" << to_hex(cpu_hash, 8) << "...)\n";
  } catch (const std::exception& e) {
    std::cerr << "[cuda] init/self-check error: " << e.what() << "\n";
    return false;
  }

  ready_ = true;
  return true;
}

void CudaMiner::start() {
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
    if (use[i] < 0 || use[i] >= n) {
      std::cerr << "[cuda] skip invalid device index " << use[i] << "\n";
      continue;
    }
    workers_.emplace_back([this, dev = use[i], i] { worker(dev, static_cast<int>(i)); });
  }
  std::cout << "[cuda] mining on " << workers_.size() << " GPU(s)\n";
}

void CudaMiner::stop() {
  stop_ = true;
  for (auto& t : workers_)
    if (t.joinable()) t.join();
  workers_.clear();
}

double CudaMiner::hashrate() const {
  uint64_t ms = now_ms() - start_ms_;
  if (!ms) return 0;
  return static_cast<double>(hashes_.load()) * 1000.0 / static_cast<double>(ms);
}

void CudaMiner::worker(int device_id, int logical_id) {
  try {
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    std::cout << "[cuda] mining on device " << device_id << " (" << prop.name << ")\n";

    // ~262k threads × 64 iters ≈ 16.7M hashes / dispatch
    const int BLOCK = 256;
    const int GRID = 1024;
    const uint32_t ITERS = 64;
    const uint64_t PER_DISPATCH =
        static_cast<uint64_t>(GRID) * BLOCK * ITERS;

    uint32_t *d_header = nullptr, *d_target = nullptr, *d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_header, 23 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_target, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result, 11 * sizeof(uint32_t)));

    // Per-job cursor + last submitted nonce so fee/user flips never re-scan
    // and re-submit the same share for a job_id.
    std::unordered_map<std::string, uint64_t> cursor_by_job;
    std::unordered_map<std::string, uint64_t> last_by_job;

    while (!stop_) {
      Job job = jobs_.get();
      if (job.job_id.empty() || job.epoch == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      const bool fee = job.is_devfee;
      const uint64_t hi = job.extranonce_hi;
      const std::string key = job.job_id + (fee ? "#fee" : "#user");

      auto it = cursor_by_job.find(key);
      if (it == cursor_by_job.end()) {
        cursor_by_job[key] = hi | (static_cast<uint64_t>(logical_id) << 40);
        last_by_job[key] = ~uint64_t{0};
      }
      uint64_t& cursor = cursor_by_job[key];
      uint64_t& last_submitted = last_by_job[key];

      uint32_t words[23] = {};
      words_from_bytes(job.blob.data(), 92, words);
      words[11] = 0;
      words[12] = 0;
      uint32_t twords[8] = {};
      words_from_bytes(job.target.data(), 32, twords);

      CUDA_CHECK(cudaMemcpy(d_header, words, sizeof(words), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_target, twords, sizeof(twords), cudaMemcpyHostToDevice));

      uint32_t zero[11] = {};
      CUDA_CHECK(cudaMemcpy(d_result, zero, sizeof(zero), cudaMemcpyHostToDevice));

      uint32_t base_lo = static_cast<uint32_t>(cursor);
      uint32_t base_hi = static_cast<uint32_t>(cursor >> 32);
      search_kernel<<<GRID, BLOCK>>>(d_header, d_target, base_lo, base_hi, ITERS, d_result);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      uint32_t res[11] = {};
      CUDA_CHECK(cudaMemcpy(res, d_result, sizeof(res), cudaMemcpyDeviceToHost));
      hashes_.fetch_add(PER_DISPATCH, std::memory_order_relaxed);

      if (res[0]) {
        uint64_t nonce = (static_cast<uint64_t>(res[2]) << 32) | res[1];
        if (nonce != last_submitted) {
          uint8_t header[92];
          std::memcpy(header, job.blob.data(), 92);
          for (int i = 0; i < 8; ++i)
            header[44 + i] = static_cast<uint8_t>((nonce >> (8 * i)) & 0xff);
          uint8_t hash[32];
          blake3_header(header, hash);
          if (hash_meets_target(hash, job.target.data())) {
            Share s;
            s.job_id = job.job_id;
            s.nonce = nonce;
            std::memcpy(s.hash.data(), hash, 32);
            s.job_epoch = job.epoch;
            s.is_devfee = fee;
            std::cout << (fee ? "[cuda/fee] " : "[cuda] ") << "gpu" << device_id
                      << " share nonce=" << to_hex(header + 44, 8)
                      << " hash=" << to_hex(hash, 8) << "...\n";
            router_.submit(s);
            last_submitted = nonce;
          } else {
            std::cerr << "[cuda] GPU hit failed CPU verify (ignored)\n";
          }
        }
        cursor = ((nonce + 1) & 0x0000FFFFFFFFFFFFULL) | hi;
      } else {
        cursor = ((cursor + PER_DISPATCH) & 0x0000FFFFFFFFFFFFULL) | hi;
      }
    }

    cudaFree(d_header);
    cudaFree(d_target);
    cudaFree(d_result);
  } catch (const std::exception& e) {
    std::cerr << "[cuda] worker " << device_id << " error: " << e.what() << "\n";
  }
}

}  // namespace alpha
