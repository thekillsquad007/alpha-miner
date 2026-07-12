#pragma once

#include "job.hpp"
#include "stratum.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace alpha {

// NVIDIA CUDA multi-GPU backend for blake3-an.
// One host thread + CUDA stream per selected device.
class CudaMiner {
 public:
  CudaMiner(JobMux& jobs, ShareRouter& router, const std::vector<int>& devices);
  ~CudaMiner();

  bool init();
  void start();
  void stop();
  double hashrate() const;

  static int device_count();
  static std::vector<std::string> list_devices();

 private:
  void worker(int device_id, int logical_id);

  JobMux& jobs_;
  ShareRouter& router_;
  std::vector<int> devices_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> hashes_{0};
  std::vector<std::thread> workers_;
  uint64_t start_ms_ = 0;
  bool ready_ = false;
};

}  // namespace alpha
