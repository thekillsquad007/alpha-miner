#pragma once

#include "core_types.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace alpha {

// AMD HIP backend (blake3-an today; sha3d uses OpenCL on AMD if HIP path not selected).
class HipMiner {
 public:
  HipMiner(miner::JobMux& jobs, miner::IShareSink& sink, const std::vector<int>& devices,
           miner::AlgoId algo = miner::AlgoId::Blake3An);
  ~HipMiner();

  bool init();
  void start();
  void stop();
  double hashrate() const;

  static int device_count();
  static std::vector<std::string> list_devices();

 private:
  void worker(int device_id, int logical_id);

  miner::JobMux& jobs_;
  miner::IShareSink& sink_;
  std::vector<int> devices_;
  miner::AlgoId algo_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> hashes_{0};
  std::vector<std::thread> workers_;
  uint64_t start_ms_ = 0;
  bool ready_ = false;
};

}  // namespace alpha
