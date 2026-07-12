#include "util.hpp"
#include "blake3.h"
#include <cstdio>
#include <chrono>
#include <stdexcept>

namespace alpha {

std::string to_hex(const uint8_t* data, size_t len) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.resize(len * 2);
  for (size_t i = 0; i < len; ++i) {
    out[i * 2] = kHex[data[i] >> 4];
    out[i * 2 + 1] = kHex[data[i] & 0xf];
  }
  return out;
}

std::vector<uint8_t> from_hex(const std::string& hex) {
  if (hex.size() % 2) throw std::runtime_error("odd hex length");
  std::vector<uint8_t> out(hex.size() / 2);
  auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::runtime_error("bad hex char");
  };
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<uint8_t>((nibble(hex[i * 2]) << 4) | nibble(hex[i * 2 + 1]));
  }
  return out;
}

bool hash_meets_target(const uint8_t hash[32], const uint8_t target[32]) {
  // Big-endian compare: hash <= target
  for (int i = 0; i < 32; ++i) {
    if (hash[i] < target[i]) return true;
    if (hash[i] > target[i]) return false;
  }
  return true;
}

uint64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

std::string format_hashrate(double hps) {
  char buf[64];
  if (hps >= 1e12) std::snprintf(buf, sizeof(buf), "%.2f TH/s", hps / 1e12);
  else if (hps >= 1e9) std::snprintf(buf, sizeof(buf), "%.2f GH/s", hps / 1e9);
  else if (hps >= 1e6) std::snprintf(buf, sizeof(buf), "%.2f MH/s", hps / 1e6);
  else if (hps >= 1e3) std::snprintf(buf, sizeof(buf), "%.2f kH/s", hps / 1e3);
  else std::snprintf(buf, sizeof(buf), "%.0f H/s", hps);
  return buf;
}

void blake3_header(const uint8_t header[92], uint8_t out[32]) {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);
  blake3_hasher_update(&hasher, header, 92);
  blake3_hasher_finalize(&hasher, out, 32);
}

}  // namespace alpha
