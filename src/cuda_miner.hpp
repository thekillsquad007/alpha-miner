#pragma once

#include "core_types.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace alpha {

// NVIDIA CUDA multi-GPU backend (blake3-an and sha3d).
class CudaMiner {
 public:
  CudaMiner(miner::JobMux& jobs, miner::IShareSink& sink, const std::vector<int>& devices,
            miner::AlgoId algo);
  ~CudaMiner();

  bool init();
  void start();
  void stop();
  double hashrate() const;

  static int device_count();
  static std::vector<std::string> list_devices();

 private:
  void worker(int device_id, int logical_id);
  void worker_blake3(int device_id, int logical_id);
  void worker_sha3d(int device_id, int logical_id);
  bool init_blake3();
  bool init_sha3d();

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
