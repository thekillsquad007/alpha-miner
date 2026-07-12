#pragma once
#include "job.hpp"
#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace alpha {

class StratumClient {
 public:
  StratumClient(std::string host, uint16_t port, std::string user, std::string pass,
                std::string agent = "alpha-miner/0.2.0", bool is_devfee = false);
  ~StratumClient();

  // Writes jobs into board (sets job.is_devfee from is_devfee_).
  void start(JobBoard& board);
  void stop();
  void submit(const Share& share);
  bool connected() const { return connected_.load(); }
  bool is_devfee() const { return is_devfee_; }
  const std::string& user() const { return user_; }

 private:
  void run();
  bool connect_once();
  void handle_line(const std::string& line, JobBoard& board);
  void send_json(const std::string& json);

  std::string host_, user_, pass_, agent_;
  uint16_t port_;
  bool is_devfee_ = false;
  int fd_ = -1;
  std::string session_id_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> connected_{false};
  std::thread thr_;
  JobBoard* board_ = nullptr;
  std::mutex send_mu_;
  std::mutex submit_mu_;
};

// Routes shares to user vs fee stratum sessions.
class ShareRouter {
 public:
  ShareRouter(StratumClient& user, StratumClient* fee) : user_(user), fee_(fee) {}
  void submit(const Share& share) {
    if (share.is_devfee && fee_) {
      fee_->submit(share);
    } else {
      user_.submit(share);
    }
  }

 private:
  StratumClient& user_;
  StratumClient* fee_;
};

}  // namespace alpha
