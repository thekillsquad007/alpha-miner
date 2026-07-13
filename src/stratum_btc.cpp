#include "stratum_btc.hpp"
#include "sha3.hpp"
#include "util.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_t = int;
static bool g_wsa_inited_btc = false;
static void ensure_wsa_btc() {
  if (!g_wsa_inited_btc) {
    WSADATA wsa{};
    WSAStartup(MAKEWORD(2, 2), &wsa);
    g_wsa_inited_btc = true;
  }
}
#define CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#define CLOSESOCK close
using SOCKET = int;
#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif
#endif

#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>

namespace miner {
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

// Extract first JSON string after position.
bool next_json_string(const std::string& j, size_t& pos, std::string& out) {
  while (pos < j.size() && j[pos] != '"') ++pos;
  if (pos >= j.size()) return false;
  ++pos;
  size_t start = pos;
  while (pos < j.size()) {
    if (j[pos] == '\\' && pos + 1 < j.size()) {
      pos += 2;
      continue;
    }
    if (j[pos] == '"') break;
    ++pos;
  }
  if (pos >= j.size()) return false;
  out = j.substr(start, pos - start);
  ++pos;
  return true;
}

// Parse mining.notify params array loosely.
bool parse_notify(const std::string& line, std::string& job_id, std::string& prevhash,
                  std::string& coinb1, std::string& coinb2, std::vector<std::string>& merkle,
                  std::string& version, std::string& nbits, std::string& ntime, bool& clean) {
  auto p = line.find("\"params\"");
  if (p == std::string::npos) return false;
  p = line.find('[', p);
  if (p == std::string::npos) return false;
  ++p;
  if (!next_json_string(line, p, job_id)) return false;
  if (!next_json_string(line, p, prevhash)) return false;
  if (!next_json_string(line, p, coinb1)) return false;
  if (!next_json_string(line, p, coinb2)) return false;

  // merkle branch array
  merkle.clear();
  auto mb = line.find('[', p);
  if (mb == std::string::npos) return false;
  auto mb_end = line.find(']', mb);
  if (mb_end == std::string::npos) return false;
  size_t q = mb + 1;
  while (q < mb_end) {
    std::string h;
    if (!next_json_string(line, q, h)) break;
    if (!h.empty()) merkle.push_back(h);
  }
  p = mb_end + 1;
  if (!next_json_string(line, p, version)) return false;
  if (!next_json_string(line, p, nbits)) return false;
  if (!next_json_string(line, p, ntime)) return false;
  // clean_jobs bool
  clean = true;
  auto tpos = line.find("true", p);
  auto fpos = line.find("false", p);
  if (fpos != std::string::npos && (tpos == std::string::npos || fpos < tpos)) clean = false;
  return true;
}

// Bitcoin stratum prevhash is 8×uint32 reversed groups of the internal LE hash.
// Convert stratum prevhash hex → 32 internal header bytes.
std::vector<uint8_t> stratum_prevhash_to_header(const std::string& hex) {
  auto raw = alpha::from_hex(hex);
  if (raw.size() != 32) return raw;
  // Each 4-byte word is byte-reversed relative to internal uint256 LE storage.
  std::vector<uint8_t> out(32);
  for (int w = 0; w < 8; ++w) {
    out[w * 4 + 0] = raw[w * 4 + 3];
    out[w * 4 + 1] = raw[w * 4 + 2];
    out[w * 4 + 2] = raw[w * 4 + 1];
    out[w * 4 + 3] = raw[w * 4 + 0];
  }
  return out;
}

uint32_t hex_u32_be(const std::string& h) {
  auto b = alpha::from_hex(h);
  uint32_t v = 0;
  // Stratum sends 8 hex chars as big-endian u32.
  for (size_t i = 0; i < b.size() && i < 4; ++i) v = (v << 8) | b[i];
  return v;
}

void write_u32_le(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xff);
  p[1] = (uint8_t)((v >> 8) & 0xff);
  p[2] = (uint8_t)((v >> 16) & 0xff);
  p[3] = (uint8_t)((v >> 24) & 0xff);
}

// Lattica merkle: SHA3-256(left || right). Leaves are txids (SHA3-256 of tx).
std::array<uint8_t, 32> sha3_merkle_root(std::array<uint8_t, 32> coinbase_hash,
                                         const std::vector<std::string>& branch) {
  std::array<uint8_t, 32> h = coinbase_hash;
  for (const auto& br_hex : branch) {
    auto br = alpha::from_hex(br_hex);
    if (br.size() != 32) continue;
    // Stratum merkle branches are typically in internal byte order already for BTC;
    // Lattica pools following BTC stratum should match. Hash(left||right) with LE bytes.
    uint8_t cat[64];
    std::memcpy(cat, h.data(), 32);
    std::memcpy(cat + 32, br.data(), 32);
    miner::sha3_256(cat, 64, h.data());
  }
  return h;
}

std::string to_hex_u32_be(uint32_t v) {
  char buf[9];
  std::snprintf(buf, sizeof(buf), "%08x", v);
  return buf;
}

void sleep_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

}  // namespace

BtcStratumClient::BtcStratumClient(std::string host, uint16_t port, std::string user,
                                   std::string pass, std::string agent, AlgoId algo)
    : host_(std::move(host)),
      user_(std::move(user)),
      pass_(std::move(pass)),
      agent_(std::move(agent)),
      port_(port),
      algo_(algo) {
#ifdef _WIN32
  ensure_wsa_btc();
#endif
}

BtcStratumClient::~BtcStratumClient() { stop(); }

void BtcStratumClient::start(JobBoard& board) {
  board_ = &board;
  stop_ = false;
  thr_ = std::thread([this] { run(); });
}

void BtcStratumClient::stop() {
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

bool BtcStratumClient::connect_once() {
#ifdef _WIN32
  ensure_wsa_btc();
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

void BtcStratumClient::send_line(const std::string& json) {
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

void BtcStratumClient::submit(const Share& share) {
  if (fd_ < 0) return;
  // mining.submit [worker, job_id, extranonce2, ntime, nonce]
  std::string nonce_hex = to_hex_u32_be(static_cast<uint32_t>(share.nonce));
  std::string ntime = share.ntime_hex.empty() ? ntime_hex_ : share.ntime_hex;
  std::string en2 = share.extranonce2_hex.empty() ? "" : share.extranonce2_hex;
  std::ostringstream oss;
  oss << "{\"id\":4,\"method\":\"mining.submit\",\"params\":[\""
      << json_escape(user_) << "\",\"" << json_escape(share.job_id) << "\",\""
      << en2 << "\",\"" << ntime << "\",\"" << nonce_hex << "\"]}";
  send_line(oss.str());
  std::cout << "[btc-stratum] submit job=" << share.job_id << " nonce=" << nonce_hex
            << " en2=" << en2 << std::endl;
}

// Skip a JSON value starting at pos (string / array / object / number / literal).
// Returns index just past the value, or npos on failure.
static size_t skip_json_value(const std::string& j, size_t pos) {
  while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t' || j[pos] == '\n' || j[pos] == '\r'))
    ++pos;
  if (pos >= j.size()) return std::string::npos;
  if (j[pos] == '"') {
    ++pos;
    while (pos < j.size()) {
      if (j[pos] == '\\' && pos + 1 < j.size()) {
        pos += 2;
        continue;
      }
      if (j[pos] == '"') return pos + 1;
      ++pos;
    }
    return std::string::npos;
  }
  if (j[pos] == '[' || j[pos] == '{') {
    char open = j[pos], close = (open == '[') ? ']' : '}';
    int depth = 0;
    bool in_str = false;
    for (; pos < j.size(); ++pos) {
      char c = j[pos];
      if (in_str) {
        if (c == '\\' && pos + 1 < j.size()) {
          ++pos;
          continue;
        }
        if (c == '"') in_str = false;
        continue;
      }
      if (c == '"') {
        in_str = true;
        continue;
      }
      if (c == open) ++depth;
      else if (c == close) {
        --depth;
        if (depth == 0) return pos + 1;
      }
    }
    return std::string::npos;
  }
  // number / true / false / null
  while (pos < j.size() && j[pos] != ',' && j[pos] != ']' && j[pos] != '}') ++pos;
  return pos;
}

static bool is_hex_string(const std::string& s) {
  if (s.empty() || (s.size() % 2) != 0) return false;
  for (char c : s) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
      return false;
  }
  return true;
}

// Parse mining.subscribe result: [ subscriptions, "extranonce1", extranonce2_size ]
static bool parse_subscribe_result(const std::string& line, std::string& en1, int& en2_size) {
  auto r = line.find("\"result\"");
  if (r == std::string::npos) return false;
  auto arr = line.find('[', r);
  if (arr == std::string::npos) return false;
  // Skip first element (subscriptions array/object)
  size_t p = arr + 1;
  p = skip_json_value(line, p);
  if (p == std::string::npos) return false;
  while (p < line.size() && (line[p] == ',' || line[p] == ' ' || line[p] == '\t')) ++p;
  if (!next_json_string(line, p, en1)) return false;
  if (!is_hex_string(en1)) return false;
  while (p < line.size() && (line[p] == ',' || line[p] == ' ' || line[p] == '\t')) ++p;
  if (p < line.size() && (line[p] >= '0' && line[p] <= '9')) {
    en2_size = std::atoi(line.c_str() + p);
    if (en2_size <= 0 || en2_size > 16) en2_size = 4;
  } else {
    en2_size = 4;
  }
  return true;
}

void BtcStratumClient::rebuild_job(JobBoard& board) {
  try {
    std::lock_guard<std::mutex> lock(job_mu_);
    if (!have_notify_) return;
    if (extranonce1_hex_.empty() || !is_hex_string(extranonce1_hex_)) {
      std::cerr << "[btc-stratum] rebuild skipped: missing/invalid extranonce1='"
                << extranonce1_hex_ << "'\n";
      return;
    }

    // Roll extranonce2
    // (std::max) avoids Windows.h min/max macros
    std::vector<uint8_t> en2(static_cast<size_t>((std::max)(extranonce2_size_, 1)), 0);
    uint64_t c = en2_counter_++;
    for (size_t i = 0; i < en2.size(); ++i) en2[i] = (uint8_t)((c >> (8 * i)) & 0xff);

    auto coinb1 = alpha::from_hex(coinb1_hex_);
    auto coinb2 = alpha::from_hex(coinb2_hex_);
    auto en1 = alpha::from_hex(extranonce1_hex_);
    std::vector<uint8_t> coinbase;
    coinbase.reserve(coinb1.size() + en1.size() + en2.size() + coinb2.size());
    coinbase.insert(coinbase.end(), coinb1.begin(), coinb1.end());
    coinbase.insert(coinbase.end(), en1.begin(), en1.end());
    coinbase.insert(coinbase.end(), en2.begin(), en2.end());
    coinbase.insert(coinbase.end(), coinb2.begin(), coinb2.end());

    // Coinbase txid = SHA3-256(non-witness coinbase) for Lattica.
    std::array<uint8_t, 32> cb_hash{};
    sha3_256(coinbase.data(), coinbase.size(), cb_hash.data());
    auto merkle = sha3_merkle_root(cb_hash, merkle_branch_);

    auto prev = stratum_prevhash_to_header(prevhash_hex_);
    if (prev.size() != 32) {
      std::cerr << "[btc-stratum] bad prevhash\n";
      return;
    }

    uint32_t version = hex_u32_be(version_hex_);
    uint32_t nbits = hex_u32_be(nbits_hex_);
    uint32_t ntime = hex_u32_be(ntime_hex_);

    Job j;
    j.algo = algo_;
    j.target_mode = TargetMode::LittleEndianUint;
    j.header_len = 80;
    j.nonce_off = 76;
    j.nonce_bytes = 4;
    j.nonce_mask = 0xFFFFFFFFULL;
    j.nonce_fixed = 0;
    j.job_id = job_id_;
    j.extranonce2_hex = alpha::to_hex(en2.data(), en2.size());
    j.ntime_hex = ntime_hex_;
    j.share_difficulty = difficulty_;
    difficulty_to_target(difficulty_, j.target.data());

    // Assemble 80-byte header (Bitcoin layout, LE fields).
    write_u32_le(j.header.data() + 0, version);
    std::memcpy(j.header.data() + 4, prev.data(), 32);
    std::memcpy(j.header.data() + 36, merkle.data(), 32);
    write_u32_le(j.header.data() + 68, ntime);
    write_u32_le(j.header.data() + 72, nbits);
    write_u32_le(j.header.data() + 76, 0);  // nonce

    board.set(j);
    std::cout << "[btc-stratum] job=" << job_id_ << " diff=" << difficulty_
              << " en2=" << j.extranonce2_hex << " branches=" << merkle_branch_.size()
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[btc-stratum] rebuild_job failed: " << e.what() << "\n";
  }
}

void BtcStratumClient::handle_line(const std::string& line, JobBoard& board) {
  // Only treat as server method push when a "method" field is present.
  // Subscribe *results* embed "mining.notify"/"mining.set_difficulty" as
  // subscription names and must not be handled here.
  const bool is_method_push =
      line.find("\"method\"") != std::string::npos || line.find("\"method\" ") != std::string::npos;

  // mining.set_difficulty
  if (is_method_push && line.find("mining.set_difficulty") != std::string::npos) {
    auto p = line.find("\"params\"");
    if (p != std::string::npos) {
      p = line.find('[', p);
      if (p != std::string::npos) {
        ++p;
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
        difficulty_ = std::strtod(line.c_str() + p, nullptr);
        if (difficulty_ <= 0) difficulty_ = 1.0;
        std::cout << "[btc-stratum] difficulty=" << difficulty_ << std::endl;
        if (have_notify_) rebuild_job(board);
      }
    }
    return;
  }

  // mining.notify
  if (is_method_push && line.find("mining.notify") != std::string::npos) {
    std::string jid, prev, c1, c2, ver, bits, ntime;
    std::vector<std::string> merkle;
    bool clean = true;
    if (!parse_notify(line, jid, prev, c1, c2, merkle, ver, bits, ntime, clean)) {
      std::cerr << "[btc-stratum] failed to parse notify\n";
      return;
    }
    {
      std::lock_guard<std::mutex> lock(job_mu_);
      job_id_ = jid;
      prevhash_hex_ = prev;
      coinb1_hex_ = c1;
      coinb2_hex_ = c2;
      merkle_branch_ = std::move(merkle);
      version_hex_ = ver;
      nbits_hex_ = bits;
      ntime_hex_ = ntime;
      clean_jobs_ = clean;
      have_notify_ = true;
      if (clean) en2_counter_ = 0;
    }
    rebuild_job(board);
    return;
  }

  // subscribe result: typically result: [ [subscriptions...], "extranonce1", extranonce2_size ]
  if (line.find("\"id\":1") != std::string::npos && line.find("\"result\"") != std::string::npos) {
    std::string en1;
    int en2 = 4;
    if (parse_subscribe_result(line, en1, en2)) {
      extranonce1_hex_ = en1;
      extranonce2_size_ = en2;
      std::cout << "[btc-stratum] subscribed en1=" << extranonce1_hex_
                << " en2_size=" << extranonce2_size_ << std::endl;
      if (have_notify_ && board_) {
        // Notify may have arrived before subscribe parse completed (unlikely) — rebuild.
        rebuild_job(board);
      }
    } else if (line.find("\"error\":null") != std::string::npos ||
               line.find("\"error\": null") != std::string::npos) {
      std::cerr << "[btc-stratum] failed to parse subscribe result: " << line.substr(0, 200)
                << "\n";
    }
  }

  // authorize result
  if (line.find("\"id\":2") != std::string::npos && line.find("true") != std::string::npos) {
    std::cout << "[btc-stratum] authorized as " << user_ << std::endl;
  }

  // submit response
  if (line.find("\"id\":4") != std::string::npos) {
    if (line.find("true") != std::string::npos && line.find("error\":null") != std::string::npos)
      std::cout << "[pool] share accepted\n";
    else if (line.find("\"error\"") != std::string::npos &&
             line.find("error\":null") == std::string::npos)
      std::cout << "[pool] " << line << std::endl;
  }
}

void BtcStratumClient::run() {
  while (!stop_) {
    std::cout << "[btc-stratum] connecting " << host_ << ":" << port_ << " ...\n";
    if (!connect_once()) {
      std::cerr << "[btc-stratum] connect failed, retry in 5s\n";
      sleep_ms(5000);
      continue;
    }
    connected_ = true;
    extranonce1_hex_.clear();
    have_notify_ = false;
    // mining.subscribe
    {
      std::ostringstream oss;
      oss << "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\""
          << json_escape(agent_) << "\"]}";
      send_line(oss.str());
    }
    // mining.authorize
    {
      std::ostringstream oss;
      oss << "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\""
          << json_escape(user_) << "\",\"" << json_escape(pass_) << "\"]}";
      send_line(oss.str());
    }
    std::cout << "[btc-stratum] subscribe+authorize as " << user_ << "\n";

    std::string buf;
    char tmp[8192];
    SOCKET s = static_cast<SOCKET>(fd_);
    while (!stop_) {
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
      std::cerr << "[btc-stratum] disconnected, reconnecting in 3s\n";
      sleep_ms(3000);
    }
  }
}

}  // namespace miner
