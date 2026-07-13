#pragma once
#include "core_types.hpp"
#include <atomic>
#include <thread>
#include <vector>

namespace alpha {

class CpuMiner {
 public:
  CpuMiner(miner::JobMux& jobs, miner::IShareSink& sink, int threads);
  ~CpuMiner();
  void start();
  void stop();
  double hashrate() const;

 private:
  void worker(int id);
  miner::JobMux& jobs_;
  miner::IShareSink& sink_;
  int threads_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> hashes_{0};
  std::vector<std::thread> workers_;
  uint64_t start_ms_ = 0;
};

}  // namespace alpha
