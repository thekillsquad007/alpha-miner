#pragma once
// Modular multi-coin work unit and mining interfaces.
//
// Adding a new coin = register a CoinProfile (algo + protocol + defaults)
// and, if the algo is new, add a kernel + backend path. Host loop stays the same.

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace miner {

enum class AlgoId : uint8_t {
  Blake3An = 0,  // Alphanumeric: BLAKE3 over 92-byte header
  Sha3d = 1,     // Lattica (and similar): SHA3-256d over 80-byte header
};

enum class ProtocolId : uint8_t {
  XmrStratum = 0,  // Monero-style login/job/submit (ALPHA pools)
  BtcStratum = 1,  // Bitcoin stratum v1 mining.subscribe/notify/submit
  GbtRpc = 2,      // Solo getblocktemplate / submitblock (JSON-RPC)
};

// How to compare hash against target.
enum class TargetMode : uint8_t {
  BigEndianBytes = 0,  // ALPHA: hash[0]..hash[31] lexicographic ≤ target
  LittleEndianUint = 1,  // Bitcoin/Lattica: interpret as LE uint256 ≤ target
};

struct Job {
  AlgoId algo = AlgoId::Blake3An;
  TargetMode target_mode = TargetMode::BigEndianBytes;
  std::string job_id;
  // Header bytes (max 128). Length is header_len.
  std::array<uint8_t, 128> header{};
  size_t header_len = 92;
  std::array<uint8_t, 32> target{};
  uint64_t height = 0;
  uint64_t epoch = 0;
  bool is_devfee = false;

  // Nonce field inside header (LE).
  size_t nonce_off = 44;
  size_t nonce_bytes = 8;  // 8 ALPHA, 4 Lattica/BTC

  // Fixed high bits of the search space (pool extranonce). For 32-bit nonces, 0.
  uint64_t nonce_fixed = 0;
  // Bits the miner may freely vary (mask applied to the full nonce value).
  uint64_t nonce_mask = 0x0000FFFFFFFFFFFFULL;  // ALPHA low 48 bits

  // Bitcoin-stratum extras (needed on submit).
  std::string extranonce2_hex;
  std::string ntime_hex;
  double share_difficulty = 0.0;

  uint64_t read_nonce() const {
    uint64_t n = 0;
    for (size_t i = 0; i < nonce_bytes && (nonce_off + i) < header_len; ++i)
      n |= (uint64_t)header[nonce_off + i] << (8 * i);
    return n;
  }

  void set_nonce(uint64_t n) {
    for (size_t i = 0; i < nonce_bytes && (nonce_off + i) < header_len; ++i)
      header[nonce_off + i] = static_cast<uint8_t>((n >> (8 * i)) & 0xff);
  }

  // ALPHA helpers
  uint64_t base_nonce() const { return read_nonce(); }
  uint64_t extranonce_hi() const { return nonce_fixed; }
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

struct JobMux {
  JobBoard user;
  JobBoard fee;
  std::atomic<uint64_t> picks{0};
  int fee_percent = 0;
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
      const uint64_t period = fee_percent >= 100 ? 1 : (100 / (uint64_t)fee_percent);
      if (period > 0 && (n % period) == 0) return fee.get();
    }
    return user.get();
  }

  uint64_t epoch_load() const {
    return user.epoch.load() + fee.epoch.load() * 1000003ull;
  }
};

struct Share {
  std::string job_id;
  uint64_t nonce = 0;
  std::array<uint8_t, 32> hash{};
  uint64_t job_epoch = 0;
  bool is_devfee = false;
  // BTC stratum
  std::string extranonce2_hex;
  std::string ntime_hex;
};

// Abstract share sink (pool or solo RPC).
class IShareSink {
 public:
  virtual ~IShareSink() = default;
  virtual void submit(const Share& share) = 0;
};

// Abstract work source (starts feeding JobBoard).
class IWorkSource {
 public:
  virtual ~IWorkSource() = default;
  virtual void start(JobBoard& board) = 0;
  virtual void stop() = 0;
  virtual bool connected() const = 0;
  virtual void submit(const Share& share) = 0;
};

// Abstract miner backend.
class IMinerBackend {
 public:
  virtual ~IMinerBackend() = default;
  virtual bool init() = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual double hashrate() const = 0;
  virtual const char* name() const = 0;
};

struct CoinProfile {
  const char* id;           // "alpha", "lattica"
  const char* display;      // "Alphanumeric", "Lattica"
  AlgoId algo;
  ProtocolId protocol;
  TargetMode target_mode;
  size_t header_len;
  size_t nonce_off;
  size_t nonce_bytes;
  uint64_t default_nonce_mask;
  const char* algo_stratum_name;  // pool algo string if any
  const char* notes;
};

const CoinProfile* find_coin(const std::string& id);
std::vector<const CoinProfile*> list_coins();
const CoinProfile& default_coin();

}  // namespace miner

// Backward-compat aliases so existing ALPHA files keep compiling during migration.
namespace alpha {
using Job = miner::Job;
using JobBoard = miner::JobBoard;
using JobMux = miner::JobMux;
using Share = miner::Share;
}  // namespace alpha
