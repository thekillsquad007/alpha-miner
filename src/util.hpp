#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace alpha {

std::string to_hex(const uint8_t* data, size_t len);
std::vector<uint8_t> from_hex(const std::string& hex);
bool hash_meets_target(const uint8_t hash[32], const uint8_t target[32]);
uint64_t now_ms();
std::string format_hashrate(double hps);

// BLAKE3 of exactly 92 header bytes.
void blake3_header(const uint8_t header[92], uint8_t out[32]);

}  // namespace alpha
