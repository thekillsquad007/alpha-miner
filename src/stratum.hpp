#pragma once
#include "core_types.hpp"
#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace alpha {

// Monero-style JSON-RPC stratum (Alphanumeric / blake3-an pools).
class StratumClient : public miner::IWorkSource {
 public:
  StratumClient(std::string host, uint16_t port, std::string user, std::string pass,
                std::string agent = "alpha-miner/0.3.0", bool is_devfee = false);
  ~StratumClient() override;

  void start(miner::JobBoard& board) override;
  void stop() override;
  void submit(const miner::Share& share) override;
  bool connected() const override { return connected_.load(); }
  bool is_devfee() const { return is_devfee_; }
  const std::string& user() const { return user_; }

 private:
  void run();
  bool connect_once();
  void handle_line(const std::string& line, miner::JobBoard& board);
  void send_json(const std::string& json);

  std::string host_, user_, pass_, agent_;
  uint16_t port_;
  bool is_devfee_ = false;
  int fd_ = -1;
  std::string session_id_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> connected_{false};
  std::thread thr_;
  miner::JobBoard* board_ = nullptr;
  std::mutex send_mu_;
  std::mutex submit_mu_;
};

class ShareRouter : public miner::IShareSink {
 public:
  ShareRouter(StratumClient& user, StratumClient* fee) : user_(user), fee_(fee) {}
  void submit(const miner::Share& share) override {
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

// Generic router over IWorkSource (used when sources are abstract).
class WorkShareRouter : public miner::IShareSink {
 public:
  explicit WorkShareRouter(miner::IWorkSource& user, miner::IWorkSource* fee = nullptr)
      : user_(user), fee_(fee) {}
  void submit(const miner::Share& share) override {
    if (share.is_devfee && fee_)
      fee_->submit(share);
    else
      user_.submit(share);
  }

 private:
  miner::IWorkSource& user_;
  miner::IWorkSource* fee_;
};

}  // namespace alpha
