#include "stratum.hpp"
#include "util.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socklen_t = int;
static bool g_wsa_inited = false;
static void ensure_wsa() {
  if (!g_wsa_inited) {
    WSADATA wsa{};
    WSAStartup(MAKEWORD(2, 2), &wsa);
    g_wsa_inited = true;
  }
}
#define CLOSESOCK closesocket
#define SOCK_ERR SOCKET_ERROR
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#define CLOSESOCK close
#define SOCK_ERR (-1)
using SOCKET = int;
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#endif

#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <chrono>

// Minimal JSON helpers for the small monero-style subset we need.

namespace alpha {
namespace {

std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '"' || c == '\\') {
      o.push_back('\\');
      o.push_back(c);
    } else {
      o.push_back(c);
    }
  }
  return o;
}

bool extract_string(const std::string& j, const std::string& key, std::string& out) {
  const std::string pat = "\"" + key + "\"";
  auto p = j.find(pat);
  if (p == std::string::npos) return false;
  p = j.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  p = j.find('"', p + 1);
  if (p == std::string::npos) return false;
  size_t start = p + 1;
  size_t end = start;
  while (end < j.size()) {
    if (j[end] == '\\' && end + 1 < j.size()) {
      end += 2;
      continue;
    }
    if (j[end] == '"') break;
    ++end;
  }
  if (end >= j.size()) return false;
  out = j.substr(start, end - start);
  return true;
}

bool extract_u64(const std::string& j, const std::string& key, uint64_t& out) {
  const std::string pat = "\"" + key + "\"";
  auto p = j.find(pat);
  if (p == std::string::npos) return false;
  p = j.find(':', p + pat.size());
  if (p == std::string::npos) return false;
  ++p;
  while (p < j.size() && (j[p] == ' ' || j[p] == '\t')) ++p;
  if (p < j.size() && j[p] == '"') {
    std::string s;
    if (!extract_string(j, key, s)) return false;
    out = std::strtoull(s.c_str(), nullptr, 10);
    return true;
  }
  char* end = nullptr;
  out = std::strtoull(j.c_str() + p, &end, 10);
  return end != j.c_str() + p;
}

bool looks_like_job(const std::string& j) {
  return j.find("\"blob\"") != std::string::npos && j.find("\"target\"") != std::string::npos;
}

void sleep_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

}  // namespace

StratumClient::StratumClient(std::string host, uint16_t port, std::string user, std::string pass,
                             std::string agent, bool is_devfee)
    : host_(std::move(host)),
      user_(std::move(user)),
      pass_(std::move(pass)),
      agent_(std::move(agent)),
      port_(port),
      is_devfee_(is_devfee) {
#ifdef _WIN32
  ensure_wsa();
#endif
}

StratumClient::~StratumClient() { stop(); }

void StratumClient::start(JobBoard& board) {
  board_ = &board;
  stop_ = false;
  thr_ = std::thread([this] { run(); });
}

void StratumClient::stop() {
  stop_ = true;
  if (fd_ >= 0) {
#ifdef _WIN32
    ::shutdown(static_cast<SOCKET>(fd_), SD_BOTH);
#else
    ::shutdown(fd_, SHUT_RDWR);
#endif
    CLOSESOCK(static_cast<SOCKET>(fd_));
    fd_ = -1;
  }
  if (thr_.joinable()) thr_.join();
  connected_ = false;
}

bool StratumClient::connect_once() {
#ifdef _WIN32
  ensure_wsa();
#endif
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = AF_UNSPEC;
  addrinfo* res = nullptr;
  std::string port_s = std::to_string(port_);
  if (getaddrinfo(host_.c_str(), port_s.c_str(), &hints, &res) != 0) return false;
  SOCKET fd = INVALID_SOCKET;
  for (auto* p = res; p; p = p->ai_next) {
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd == INVALID_SOCKET) continue;
    if (::connect(fd, p->ai_addr, (int)p->ai_addrlen) == 0) break;
    CLOSESOCK(fd);
    fd = INVALID_SOCKET;
  }
  freeaddrinfo(res);
  if (fd == INVALID_SOCKET) return false;
  fd_ = static_cast<int>(fd);
  return true;
}

void StratumClient::send_json(const std::string& json) {
  std::lock_guard<std::mutex> lock(send_mu_);
  if (fd_ < 0) return;
  std::string msg = json + "\n";
  size_t off = 0;
  SOCKET s = static_cast<SOCKET>(fd_);
  while (off < msg.size()) {
    int n = ::send(s, msg.data() + off, (int)(msg.size() - off), 0);
    if (n <= 0) break;
    off += static_cast<size_t>(n);
  }
}

void StratumClient::submit(const Share& share) {
  std::lock_guard<std::mutex> lock(submit_mu_);
  if (session_id_.empty() || fd_ < 0) return;
  uint8_t nb[8];
  for (int i = 0; i < 8; ++i) nb[i] = static_cast<uint8_t>((share.nonce >> (8 * i)) & 0xff);
  std::string nonce_hex = to_hex(nb, 8);
  std::string result_hex = to_hex(share.hash.data(), 32);
  std::ostringstream oss;
  oss << "{\"id\":2,\"jsonrpc\":\"2.0\",\"method\":\"submit\",\"params\":{"
      << "\"id\":\"" << json_escape(session_id_) << "\","
      << "\"job_id\":\"" << json_escape(share.job_id) << "\","
      << "\"nonce\":\"" << nonce_hex << "\","
      << "\"result\":\"" << result_hex << "\"}}";
  send_json(oss.str());
  std::cout << (is_devfee_ || share.is_devfee ? "[submit/fee] " : "[submit] ")
            << "job=" << share.job_id << " nonce=" << nonce_hex << std::endl;
}

void StratumClient::handle_line(const std::string& line, JobBoard& board) {
  if (line.find("\"status\":\"OK\"") != std::string::npos ||
      (line.find("\"result\"") != std::string::npos && line.find("\"job\"") != std::string::npos)) {
    auto r = line.find("\"result\"");
    if (r != std::string::npos) {
      auto idp = line.find("\"id\"", r);
      if (idp != std::string::npos) {
        std::string tmp;
        extract_string(line.substr(idp), "id", tmp);
        if (!tmp.empty() && tmp.size() < 16) session_id_ = tmp;
      }
    }
  }

  if (line.find("\"method\":\"job\"") != std::string::npos || looks_like_job(line)) {
    std::string blob_hex, target_hex, job_id;
    uint64_t height = 0;
    if (!extract_string(line, "blob", blob_hex)) return;
    if (!extract_string(line, "target", target_hex)) return;
    extract_string(line, "job_id", job_id);
    extract_u64(line, "height", height);
    auto blob = from_hex(blob_hex);
    auto target = from_hex(target_hex);
    if (blob.size() != 92 || target.size() != 32) {
      std::cerr << "[stratum] bad blob/target size\n";
      return;
    }
    Job j;
    j.job_id = job_id;
    j.height = height;
    j.is_devfee = is_devfee_;
    std::copy(blob.begin(), blob.end(), j.blob.begin());
    std::copy(target.begin(), target.end(), j.target.begin());
    uint64_t base = j.base_nonce();
    j.extranonce_hi = base & 0xFFFF000000000000ULL;
    board.set(j);
    std::cout << (is_devfee_ ? "[job/fee] " : "[job] ") << "id=" << job_id
              << " height=" << height << " extranonce_hi=0x" << std::hex
              << (j.extranonce_hi >> 48) << std::dec << std::endl;
  }

  if (line.find("\"method\":\"job\"") == std::string::npos) {
    if (line.find("\"error\":null") != std::string::npos &&
        line.find("\"result\":true") != std::string::npos) {
      std::cout << "[pool] share accepted\n";
    } else if (line.find("\"error\"") != std::string::npos &&
               line.find("\"error\":null") == std::string::npos &&
               line.find("\"id\":2") != std::string::npos) {
      std::cout << "[pool] " << line << std::endl;
    }
  }

  if (line.find("\"extensions\"") != std::string::npos) {
    auto r = line.find("\"result\"");
    if (r != std::string::npos) {
      std::string sid;
      if (extract_string(line.substr(r), "id", sid)) session_id_ = sid;
    }
  }
}

void StratumClient::run() {
  while (!stop_) {
    std::cout << "[stratum] connecting " << host_ << ":" << port_ << " ...\n";
    if (!connect_once()) {
      std::cerr << "[stratum] connect failed, retry in 5s\n";
      sleep_ms(5000);
      continue;
    }
    connected_ = true;
    std::ostringstream login;
    login << "{\"id\":1,\"jsonrpc\":\"2.0\",\"method\":\"login\",\"params\":{"
          << "\"login\":\"" << json_escape(user_) << "\","
          << "\"pass\":\"" << json_escape(pass_) << "\","
          << "\"agent\":\"" << json_escape(agent_) << "\","
          << "\"algo\":[\"blake3-an\"]}}";
    send_json(login.str());
    std::cout << "[stratum] login as " << user_ << "\n";

    std::string buf;
    char tmp[4096];
    uint64_t last_ping = now_ms();
    SOCKET s = static_cast<SOCKET>(fd_);
    while (!stop_) {
      if (now_ms() - last_ping > 60000) {
        send_json("{\"id\":3,\"jsonrpc\":\"2.0\",\"method\":\"keepalived\",\"params\":{}}");
        last_ping = now_ms();
      }
      fd_set rfds;
      FD_ZERO(&rfds);
      FD_SET(s, &rfds);
      timeval tv{1, 0};
      int rv = ::select(static_cast<int>(s) + 1, &rfds, nullptr, nullptr, &tv);
      if (rv < 0) break;
      if (rv == 0) continue;
      int n = ::recv(s, tmp, sizeof(tmp), 0);
      if (n <= 0) break;
      buf.append(tmp, tmp + n);
      size_t pos;
      while ((pos = buf.find('\n')) != std::string::npos) {
        std::string line = buf.substr(0, pos);
        buf.erase(0, pos + 1);
        if (!line.empty() && board_) handle_line(line, *board_);
      }
    }
    connected_ = false;
    if (fd_ >= 0) {
      CLOSESOCK(static_cast<SOCKET>(fd_));
      fd_ = -1;
    }
    if (!stop_) {
      std::cerr << "[stratum] disconnected, reconnecting in 3s\n";
      sleep_ms(3000);
    }
  }
}

}  // namespace alpha
