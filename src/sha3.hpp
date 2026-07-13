#pragma once
// Portable SHA3-256 (FIPS 202) + SHA3-256d for Lattica PoW.
// Byte-identical to Lattica Core crypto/sha3.cpp + CBlockHeader::GetPoWHash.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace miner {

// Single SHA3-256 of arbitrary message.
void sha3_256(const uint8_t* data, size_t len, uint8_t out[32]);

// Lattica PoW: SHA3-256(SHA3-256(header[80])).
void sha3_256d_header80(const uint8_t header[80], uint8_t out[32]);

// LE uint256 compare: hash ≤ target (Bitcoin/Lattica style).
bool hash_meets_target_le(const uint8_t hash[32], const uint8_t target[32]);

// Compact nBits → 32-byte LE target (Bitcoin compact encoding).
bool compact_to_target(uint32_t nbits, uint8_t target[32]);

// Share difficulty (Bitcoin-style) → 32-byte LE target.
// Uses the classic diff-1 target 0x00000000FFFF0000... (same as most SHA3d pools / ccminer).
void difficulty_to_target(double difficulty, uint8_t target[32]);

// Self-test against FIPS-202 vectors. Returns true on success.
bool sha3_selftest();

}  // namespace miner
