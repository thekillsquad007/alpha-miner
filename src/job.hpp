#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace alpha {

// 92-byte alphanumeric mining header (LE fields):
// index u32 | prev_hash 32 | timestamp u64 | nonce u64 | difficulty u64 | merkle 32
struct Job {
  std::string job_id;
  std::array<uint8_t, 92> blob{};
  std::array<uint8_t, 32> target{};
  uint64_t height = 0;
  uint64_t extranonce_hi = 0;  // top 16 bits of nonce (pool-bound)
  uint64_t epoch = 0;          // bumps on every new job
  bool is_devfee = false;      // true → submit on fee stratum session

  uint64_t base_nonce() const {
    uint64_t n = 0;
    for (int i = 0; i < 8; ++i) n |= (uint64_t)blob[44 + i] << (8 * i);
    return n;
  }

  void set_nonce(uint64_t n) {
    for (int i = 0; i < 8; ++i) blob[44 + i] = static_cast<uint8_t>((n >> (8 * i)) & 0xff);
  }
};

struct JobBoard {
  mutable std::mutex mu;
  Job current;
  std::atomic<uint64_t> epoch{0};

  void set(Job j) {
    std::lock_guard<std::mutex> lock(mu);
    j.epoch = epoch.fetch_add(1) + 1;
    current = std::move(j);
  }

  Job get() const {
    std::lock_guard<std::mutex> lock(mu);
    return current;
  }

  bool has_job() const {
    std::lock_guard<std::mutex> lock(mu);
    return !current.job_id.empty() && current.epoch != 0;
  }
};

// Merges miner + optional 2% developer-fee job stream.
// Roughly kPercent of returned templates are fee jobs (when available).
struct JobMux {
  JobBoard user;
  JobBoard fee;
  std::atomic<uint64_t> picks{0};
  int fee_percent = 2;  // 0..100
  bool fee_enabled = false;

  void set_user(Job j) {
    j.is_devfee = false;
    user.set(std::move(j));
  }
  void set_fee(Job j) {
    j.is_devfee = true;
    fee.set(std::move(j));
  }

  Job get() {
    const uint64_t n = picks.fetch_add(1, std::memory_order_relaxed);
    if (fee_enabled && fee_percent > 0 && fee.has_job()) {
      // e.g. 2% → every 50th pick (n % 50 == 0)
      const uint64_t period = fee_percent >= 100 ? 1 : (100 / (uint64_t)fee_percent);
      if (period > 0 && (n % period) == 0) {
        return fee.get();
      }
    }
    return user.get();
  }

  // Compatibility with code that polled board.epoch for tip changes.
  std::atomic<uint64_t>& epoch_ref() { return user.epoch; }
  uint64_t epoch_load() const {
    // Change if either side updates — miners re-read get().
    return user.epoch.load() + fee.epoch.load() * 1000003ull;
  }
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  std::array<uint8_t, 32> hash{};
  uint64_t job_epoch = 0;
  bool is_devfee = false;
};

}  // namespace alpha
