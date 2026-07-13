#pragma once
#include "core_types.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace alpha {

std::string to_hex(const uint8_t* data, size_t len);
std::vector<uint8_t> from_hex(const std::string& hex);
// Big-endian byte compare (ALPHA).
bool hash_meets_target(const uint8_t hash[32], const uint8_t target[32]);
// Dispatch by job target mode.
bool hash_meets_job_target(const uint8_t hash[32], const miner::Job& job);
uint64_t now_ms();
std::string format_hashrate(double hps);

// BLAKE3 of exactly 92 header bytes.
void blake3_header(const uint8_t header[92], uint8_t out[32]);

// Hash a job header with the correct algorithm.
void hash_job_header(const miner::Job& job, const uint8_t* header, uint8_t out[32]);

}  // namespace alpha

namespace miner {
using alpha::to_hex;
using alpha::from_hex;
using alpha::now_ms;
using alpha::format_hashrate;
using alpha::hash_meets_job_target;
using alpha::hash_job_header;
}  // namespace miner
