#include "cpu_miner.hpp"
#include "util.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

namespace alpha {

CpuMiner::CpuMiner(JobMux& jobs, ShareRouter& router, int threads)
    : jobs_(jobs), router_(router), threads_(threads) {}

CpuMiner::~CpuMiner() { stop(); }

void CpuMiner::start() {
  stop_ = false;
  hashes_ = 0;
  start_ms_ = now_ms();
  workers_.clear();
  for (int i = 0; i < threads_; ++i) {
    workers_.emplace_back([this, i] { worker(i); });
  }
}

void CpuMiner::stop() {
  stop_ = true;
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
  workers_.clear();
}

double CpuMiner::hashrate() const {
  uint64_t ms = now_ms() - start_ms_;
  if (ms == 0) return 0;
  return (double)hashes_.load() * 1000.0 / (double)ms;
}

void CpuMiner::worker(int id) {
  while (!stop_) {
    Job job = jobs_.get();
    if (job.job_id.empty() || job.epoch == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    const uint64_t epoch = job.epoch;
    const bool fee = job.is_devfee;
    const uint64_t hi = job.extranonce_hi;
    uint64_t cursor = (uint64_t)id;
    uint8_t header[92];
    std::memcpy(header, job.blob.data(), 92);
    uint64_t local = 0;
    while (!stop_) {
      // Re-pick when either user or fee job stream advances, or fee slice ends.
      Job latest = jobs_.get();
      if (latest.job_id != job.job_id || latest.epoch != epoch || latest.is_devfee != fee) break;

      uint64_t nonce = hi | (cursor & 0x0000FFFFFFFFFFFFULL);
      for (int i = 0; i < 8; ++i) header[44 + i] = (uint8_t)((nonce >> (8 * i)) & 0xff);
      uint8_t hash[32];
      blake3_header(header, hash);
      ++local;
      if ((local & 0x3fff) == 0) {
        hashes_.fetch_add(0x4000, std::memory_order_relaxed);
        local = 0;
      }
      if (hash_meets_target(hash, job.target.data())) {
        hashes_.fetch_add(local, std::memory_order_relaxed);
        local = 0;
        Share s;
        s.job_id = job.job_id;
        s.nonce = nonce;
        std::memcpy(s.hash.data(), hash, 32);
        s.job_epoch = epoch;
        s.is_devfee = fee;
        std::cout << (fee ? "[cpu/fee] " : "[cpu] ") << "share nonce=" << to_hex(header + 44, 8)
                  << std::endl;
        router_.submit(s);
      }
      cursor += (uint64_t)threads_;
    }
    if (local) hashes_.fetch_add(local, std::memory_order_relaxed);
  }
}

}  // namespace alpha
