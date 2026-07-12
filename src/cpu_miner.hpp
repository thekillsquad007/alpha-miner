#pragma once
#include "job.hpp"
#include "stratum.hpp"
#include <atomic>
#include <thread>
#include <vector>

namespace alpha {

class CpuMiner {
 public:
  CpuMiner(JobMux& jobs, ShareRouter& router, int threads);
  ~CpuMiner();
  void start();
  void stop();
  double hashrate() const;

 private:
  void worker(int id);
  JobMux& jobs_;
  ShareRouter& router_;
  int threads_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> hashes_{0};
  std::vector<std::thread> workers_;
  uint64_t start_ms_ = 0;
};

}  // namespace alpha
