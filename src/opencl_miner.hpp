#pragma once
#ifdef ALPHA_HAS_OPENCL
#include "job.hpp"
#include "stratum.hpp"
#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace alpha {

class OpenClMiner {
 public:
  OpenClMiner(JobMux& jobs, ShareRouter& router, std::string kernel_path,
             const std::vector<int>& devices);
  ~OpenClMiner();
  bool init();
  void start();
  void stop();
  double hashrate() const;
  static std::vector<std::string> list_devices();

 private:
  void worker(int device_index, int logical_id);
  JobMux& jobs_;
  ShareRouter& router_;
  std::string kernel_path_;
  std::vector<int> devices_;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> hashes_{0};
  std::vector<std::thread> workers_;
  uint64_t start_ms_ = 0;
};

}  // namespace alpha
#endif
