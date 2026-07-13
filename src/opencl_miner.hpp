#pragma once
#ifdef ALPHA_HAS_OPENCL
#include "core_types.hpp"
#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace alpha {

class OpenClMiner {
 public:
  OpenClMiner(miner::JobMux& jobs, miner::IShareSink& sink, std::string kernel_dir,
             const std::vector<int>& devices, miner::AlgoId algo);
  ~OpenClMiner();
  bool init();
  void start();
  void stop();
  double hashrate() const;
  static std::vector<std::string> list_devices();

 private:
  void worker(int device_index, int logical_id);
  void worker_blake3(int device_index, int logical_id);
  void worker_sha3d(int device_index, int logical_id);

  miner::JobMux& jobs_;
  miner::IShareSink& sink_;
  std::string kernel_dir_;
  std::vector<int> devices_;
  miner::AlgoId algo_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> hashes_{0};
  std::vector<std::thread> workers_;
  uint64_t start_ms_ = 0;
};

}  // namespace alpha
#endif
