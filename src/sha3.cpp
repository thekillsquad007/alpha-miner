#include "sha3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace miner {
namespace {

inline uint64_t rotl64(uint64_t x, int n) {
  return (x << n) | (x >> (64 - n));
}

inline uint64_t read_le64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
  return v;
}

inline void write_le64(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = (uint8_t)((v >> (8 * i)) & 0xff);
}

void keccakf(uint64_t st[25]) {
  static constexpr uint64_t RNDC[24] = {
      0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
      0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
      0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
      0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
      0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
      0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

  for (int round = 0; round < 24; ++round) {
    uint64_t bc0 = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
    uint64_t bc1 = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
    uint64_t bc2 = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
    uint64_t bc3 = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
    uint64_t bc4 = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

    uint64_t t = bc4 ^ rotl64(bc1, 1);
    st[0] ^= t;
    st[5] ^= t;
    st[10] ^= t;
    st[15] ^= t;
    st[20] ^= t;
    t = bc0 ^ rotl64(bc2, 1);
    st[1] ^= t;
    st[6] ^= t;
    st[11] ^= t;
    st[16] ^= t;
    st[21] ^= t;
    t = bc1 ^ rotl64(bc3, 1);
    st[2] ^= t;
    st[7] ^= t;
    st[12] ^= t;
    st[17] ^= t;
    st[22] ^= t;
    t = bc2 ^ rotl64(bc4, 1);
    st[3] ^= t;
    st[8] ^= t;
    st[13] ^= t;
    st[18] ^= t;
    st[23] ^= t;
    t = bc3 ^ rotl64(bc0, 1);
    st[4] ^= t;
    st[9] ^= t;
    st[14] ^= t;
    st[19] ^= t;
    st[24] ^= t;

    t = st[1];
    uint64_t b0;
    b0 = st[10];
    st[10] = rotl64(t, 1);
    t = b0;
    b0 = st[7];
    st[7] = rotl64(t, 3);
    t = b0;
    b0 = st[11];
    st[11] = rotl64(t, 6);
    t = b0;
    b0 = st[17];
    st[17] = rotl64(t, 10);
    t = b0;
    b0 = st[18];
    st[18] = rotl64(t, 15);
    t = b0;
    b0 = st[3];
    st[3] = rotl64(t, 21);
    t = b0;
    b0 = st[5];
    st[5] = rotl64(t, 28);
    t = b0;
    b0 = st[16];
    st[16] = rotl64(t, 36);
    t = b0;
    b0 = st[8];
    st[8] = rotl64(t, 45);
    t = b0;
    b0 = st[21];
    st[21] = rotl64(t, 55);
    t = b0;
    b0 = st[24];
    st[24] = rotl64(t, 2);
    t = b0;
    b0 = st[4];
    st[4] = rotl64(t, 14);
    t = b0;
    b0 = st[15];
    st[15] = rotl64(t, 27);
    t = b0;
    b0 = st[23];
    st[23] = rotl64(t, 41);
    t = b0;
    b0 = st[19];
    st[19] = rotl64(t, 56);
    t = b0;
    b0 = st[13];
    st[13] = rotl64(t, 8);
    t = b0;
    b0 = st[12];
    st[12] = rotl64(t, 25);
    t = b0;
    b0 = st[2];
    st[2] = rotl64(t, 43);
    t = b0;
    b0 = st[20];
    st[20] = rotl64(t, 62);
    t = b0;
    b0 = st[14];
    st[14] = rotl64(t, 18);
    t = b0;
    b0 = st[22];
    st[22] = rotl64(t, 39);
    t = b0;
    b0 = st[9];
    st[9] = rotl64(t, 61);
    t = b0;
    b0 = st[6];
    st[6] = rotl64(t, 20);
    t = b0;
    st[1] = rotl64(t, 44);

    uint64_t c0, c1, c2, c3, c4;
    c0 = st[0];
    c1 = st[1];
    c2 = st[2];
    c3 = st[3];
    c4 = st[4];
    st[0] = c0 ^ ((~c1) & c2) ^ RNDC[round];
    st[1] = c1 ^ ((~c2) & c3);
    st[2] = c2 ^ ((~c3) & c4);
    st[3] = c3 ^ ((~c4) & c0);
    st[4] = c4 ^ ((~c0) & c1);
    c0 = st[5];
    c1 = st[6];
    c2 = st[7];
    c3 = st[8];
    c4 = st[9];
    st[5] = c0 ^ ((~c1) & c2);
    st[6] = c1 ^ ((~c2) & c3);
    st[7] = c2 ^ ((~c3) & c4);
    st[8] = c3 ^ ((~c4) & c0);
    st[9] = c4 ^ ((~c0) & c1);
    c0 = st[10];
    c1 = st[11];
    c2 = st[12];
    c3 = st[13];
    c4 = st[14];
    st[10] = c0 ^ ((~c1) & c2);
    st[11] = c1 ^ ((~c2) & c3);
    st[12] = c2 ^ ((~c3) & c4);
    st[13] = c3 ^ ((~c4) & c0);
    st[14] = c4 ^ ((~c0) & c1);
    c0 = st[15];
    c1 = st[16];
    c2 = st[17];
    c3 = st[18];
    c4 = st[19];
    st[15] = c0 ^ ((~c1) & c2);
    st[16] = c1 ^ ((~c2) & c3);
    st[17] = c2 ^ ((~c3) & c4);
    st[18] = c3 ^ ((~c4) & c0);
    st[19] = c4 ^ ((~c0) & c1);
    c0 = st[20];
    c1 = st[21];
    c2 = st[22];
    c3 = st[23];
    c4 = st[24];
    st[20] = c0 ^ ((~c1) & c2);
    st[21] = c1 ^ ((~c2) & c3);
    st[22] = c2 ^ ((~c3) & c4);
    st[23] = c3 ^ ((~c4) & c0);
    st[24] = c4 ^ ((~c0) & c1);
  }
}

}  // namespace

void sha3_256(const uint8_t* data, size_t len, uint8_t out[32]) {
  uint64_t st[25] = {};
  // Absorb full rate blocks (136 bytes = 17 lanes).
  while (len >= 136) {
    for (int i = 0; i < 17; ++i) st[i] ^= read_le64(data + 8 * i);
    keccakf(st);
    data += 136;
    len -= 136;
  }
  uint8_t buf[136] = {};
  if (len) std::memcpy(buf, data, len);
  buf[len] ^= 0x06;  // SHA3 domain
  buf[135] ^= 0x80;  // multi-rate padding
  for (int i = 0; i < 17; ++i) st[i] ^= read_le64(buf + 8 * i);
  keccakf(st);
  for (int i = 0; i < 4; ++i) write_le64(out + 8 * i, st[i]);
}

void sha3_256d_header80(const uint8_t header[80], uint8_t out[32]) {
  // Fast path: single-block absorb for 80-byte header (matches GPU midstate layout).
  uint64_t st[25] = {};
  for (int i = 0; i < 10; ++i) st[i] = read_le64(header + 8 * i);
  st[10] ^= 0x06ULL;
  st[16] ^= 0x8000000000000000ULL;
  keccakf(st);

  uint64_t a = st[0], b = st[1], c = st[2], d = st[3];
  std::memset(st, 0, sizeof(st));
  st[0] = a;
  st[1] = b;
  st[2] = c;
  st[3] = d;
  st[4] = 0x06ULL;
  st[16] = 0x8000000000000000ULL;
  keccakf(st);
  for (int i = 0; i < 4; ++i) write_le64(out + 8 * i, st[i]);
}

bool hash_meets_target_le(const uint8_t hash[32], const uint8_t target[32]) {
  // Compare as little-endian 256-bit: start from most-significant limb (bytes 24..31).
  for (int i = 31; i >= 0; --i) {
    if (hash[i] < target[i]) return true;
    if (hash[i] > target[i]) return false;
  }
  return true;
}

bool compact_to_target(uint32_t nbits, uint8_t target[32]) {
  std::memset(target, 0, 32);
  const uint32_t exp = nbits >> 24;
  uint32_t mant = nbits & 0x007fffff;
  if (nbits & 0x00800000) return false;  // negative
  if (exp <= 3) {
    mant >>= 8 * (3 - exp);
    target[0] = (uint8_t)(mant & 0xff);
    target[1] = (uint8_t)((mant >> 8) & 0xff);
    target[2] = (uint8_t)((mant >> 16) & 0xff);
  } else {
    const size_t off = exp - 3;
    if (off >= 32) return false;
    if (off + 0 < 32) target[off + 0] = (uint8_t)(mant & 0xff);
    if (off + 1 < 32) target[off + 1] = (uint8_t)((mant >> 8) & 0xff);
    if (off + 2 < 32) target[off + 2] = (uint8_t)((mant >> 16) & 0xff);
  }
  return mant != 0;
}

void difficulty_to_target(double difficulty, uint8_t target[32]) {
  // diff1 = 0x00000000FFFF0000... (Bitcoin)
  // target = diff1 / difficulty
  if (difficulty < 1e-12) difficulty = 1e-12;
  // Work in floating 256-bit approx via high 64 bits of diff1.
  // diff1_hi = 0x00000000FFFF0000 as the top 64 bits of the 256-bit value
  // shifted so that we use double math carefully.
  //
  // Represent target as (0xFFFF0000 << 192) / difficulty, then write LE bytes.
  // Using integer path: for typical pool diffs this is accurate enough.
  const double diff1 = 26959535291011309493156476344723991336010898738574164086137773096960.0;
  // That constant is 0xffff0000 * 2^192 as double (lossy at low bits, fine for share targets).
  double t = diff1 / difficulty;
  std::memset(target, 0, 32);
  // Peel off 8 bytes at a time from high to low into LE buffer.
  // We'll fill from MSB limb downward using frexp-style division by 2^64.
  // Simpler: convert via successive mod 2^32 from low end of a big float.
  for (int i = 0; i < 8; ++i) {
    double limb_d = std::fmod(t, 4294967296.0);
    if (limb_d < 0) limb_d = 0;
    uint32_t limb = (uint32_t)limb_d;
    target[i * 4 + 0] = (uint8_t)(limb & 0xff);
    target[i * 4 + 1] = (uint8_t)((limb >> 8) & 0xff);
    target[i * 4 + 2] = (uint8_t)((limb >> 16) & 0xff);
    target[i * 4 + 3] = (uint8_t)((limb >> 24) & 0xff);
    t = std::floor(t / 4294967296.0);
  }
}

bool sha3_selftest() {
  // FIPS 202: SHA3-256("") and SHA3-256("abc")
  uint8_t out[32];
  sha3_256(nullptr, 0, out);
  static const char* empty =
      "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a";
  char hex[65];
  static const char* hx = "0123456789abcdef";
  for (int i = 0; i < 32; ++i) {
    hex[i * 2] = hx[out[i] >> 4];
    hex[i * 2 + 1] = hx[out[i] & 0xf];
  }
  hex[64] = 0;
  if (std::string(hex) != empty) return false;

  const uint8_t abc[3] = {'a', 'b', 'c'};
  sha3_256(abc, 3, out);
  static const char* abch =
      "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532";
  for (int i = 0; i < 32; ++i) {
    hex[i * 2] = hx[out[i] >> 4];
    hex[i * 2 + 1] = hx[out[i] & 0xf];
  }
  return std::string(hex) == abch;
}

}  // namespace miner
