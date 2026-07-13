#pragma once
// Bitcoin stratum v1 work source — used by Lattica pools (e.g. coin-miners.info).
// mining.subscribe / authorize / notify / set_difficulty / submit

#include "core_types.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace miner {

class BtcStratumClient : public IWorkSource {
 public:
  BtcStratumClient(std::string host, uint16_t port, std::string user, std::string pass,
                   std::string agent = "alpha-miner/0.3.0-lattica",
                   AlgoId algo = AlgoId::Sha3d);
  ~BtcStratumClient() override;

  void start(JobBoard& board) override;
  void stop() override;
  void submit(const Share& share) override;
  bool connected() const override { return connected_.load(); }

 private:
  void run();
  bool connect_once();
  void send_line(const std::string& json);
  void handle_line(const std::string& line, JobBoard& board);
  void rebuild_job(JobBoard& board);

  std::string host_, user_, pass_, agent_;
  uint16_t port_;
  AlgoId algo_;
  int fd_ = -1;
  std::atomic<bool> stop_{false};
  std::atomic<bool> connected_{false};
  std::thread thr_;
  JobBoard* board_ = nullptr;
  std::mutex send_mu_;
  std::mutex job_mu_;

  // Session state
  std::string extranonce1_hex_;
  int extranonce2_size_ = 4;
  double difficulty_ = 1.0;
  uint64_t en2_counter_ = 0;

  // Current notify fields
  std::string job_id_;
  std::string prevhash_hex_;   // 64 hex, stratum order (word-swapped for BTC)
  std::string coinb1_hex_;
  std::string coinb2_hex_;
  std::vector<std::string> merkle_branch_;
  std::string version_hex_;
  std::string nbits_hex_;
  std::string ntime_hex_;
  bool clean_jobs_ = true;
  bool have_notify_ = false;
};

}  // namespace miner
