// BLAKE3 one-shot hash of the 92-byte alphanumeric header.
// Each work-item tests `iters` nonces starting at base_nonce + gid.
// Header layout (LE): index4 | prev32 | ts8 | nonce8 | diff8 | merkle32

#define IV0 0x6A09E667u
#define IV1 0xBB67AE85u
#define IV2 0x3C6EF372u
#define IV3 0xA54FF53Au
#define IV4 0x510E527Fu
#define IV5 0x9B05688Cu
#define IV6 0x1F83D9ABu
#define IV7 0x5BE0CD19u
#define CHUNK_START 1u
#define CHUNK_END 2u
#define ROOT 8u

inline uint rotr(uint x, uint n) { return (x >> n) | (x << (32u - n)); }

inline void g(uint *s, uint a, uint b, uint c, uint d, uint mx, uint my) {
  s[a] = s[a] + s[b] + mx;
  s[d] = rotr(s[d] ^ s[a], 16u);
  s[c] = s[c] + s[d];
  s[b] = rotr(s[b] ^ s[c], 12u);
  s[a] = s[a] + s[b] + my;
  s[d] = rotr(s[d] ^ s[a], 8u);
  s[c] = s[c] + s[d];
  s[b] = rotr(s[b] ^ s[c], 7u);
}

inline void compress(const uint cv[8], const uint block[16], uint block_len, uint flags, uint out[8]) {
  uint s[16];
  s[0]=cv[0]; s[1]=cv[1]; s[2]=cv[2]; s[3]=cv[3];
  s[4]=cv[4]; s[5]=cv[5]; s[6]=cv[6]; s[7]=cv[7];
  s[8]=IV0; s[9]=IV1; s[10]=IV2; s[11]=IV3;
  s[12]=0u; s[13]=0u; s[14]=block_len; s[15]=flags;

  uint m0=block[0],m1=block[1],m2=block[2],m3=block[3];
  uint m4=block[4],m5=block[5],m6=block[6],m7=block[7];
  uint m8=block[8],m9=block[9],m10=block[10],m11=block[11];
  uint m12=block[12],m13=block[13],m14=block[14],m15=block[15];

  for (uint round = 0; round < 7u; ++round) {
    g(s,0,4,8,12,m0,m1);  g(s,1,5,9,13,m2,m3);
    g(s,2,6,10,14,m4,m5); g(s,3,7,11,15,m6,m7);
    g(s,0,5,10,15,m8,m9); g(s,1,6,11,12,m10,m11);
    g(s,2,7,8,13,m12,m13); g(s,3,4,9,14,m14,m15);
    if (round < 6u) {
      uint n0=m2,n1=m6,n2=m3,n3=m10,n4=m7,n5=m0,n6=m4,n7=m13;
      uint n8=m1,n9=m11,n10=m12,n11=m5,n12=m9,n13=m14,n14=m15,n15=m8;
      m0=n0;m1=n1;m2=n2;m3=n3;m4=n4;m5=n5;m6=n6;m7=n7;
      m8=n8;m9=n9;m10=n10;m11=n11;m12=n12;m13=n13;m14=n14;m15=n15;
    }
  }
  out[0]=s[0]^s[8]; out[1]=s[1]^s[9]; out[2]=s[2]^s[10]; out[3]=s[3]^s[11];
  out[4]=s[4]^s[12]; out[5]=s[5]^s[13]; out[6]=s[6]^s[14]; out[7]=s[7]^s[15];
}

inline void blake3_92(const uint w[23], uint nonce_lo, uint nonce_hi, uint out[8]) {
  // w0..w10 fixed; w11,w12 = nonce; w13..w22 fixed. w23 pad=0
  uint cv[8] = {IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7};
  uint block0[16];
  block0[0]=w[0]; block0[1]=w[1]; block0[2]=w[2]; block0[3]=w[3];
  block0[4]=w[4]; block0[5]=w[5]; block0[6]=w[6]; block0[7]=w[7];
  block0[8]=w[8]; block0[9]=w[9]; block0[10]=w[10];
  block0[11]=nonce_lo; block0[12]=nonce_hi;
  block0[13]=w[13]; block0[14]=w[14]; block0[15]=w[15];
  uint mid[8];
  compress(cv, block0, 64u, CHUNK_START, mid);

  uint block1[16];
  block1[0]=w[16]; block1[1]=w[17]; block1[2]=w[18]; block1[3]=w[19];
  block1[4]=w[20]; block1[5]=w[21]; block1[6]=w[22];
  block1[7]=0; block1[8]=0; block1[9]=0; block1[10]=0;
  block1[11]=0; block1[12]=0; block1[13]=0; block1[14]=0; block1[15]=0;
  compress(mid, block1, 28u, CHUNK_END | ROOT, out);
}

inline int meets_target(const uint h[8], __constant uint *target) {
  // hash and target as big-endian byte strings; words are LE in memory
  for (int i = 0; i < 8; ++i) {
    uint hw = h[i];
    uint tw = target[i];
    // compare byte-by-byte from MSB of word (last byte of LE word first for BE stream)
    // Actually: hash bytes are little-endian words in BLAKE3 output = sequential bytes
    // hash byte k is in word k/4, byte k%4 (LE).
    // target is provided as 32 raw bytes loaded into uint LE words the same way.
    uchar hb[4] = {(uchar)(hw), (uchar)(hw>>8), (uchar)(hw>>16), (uchar)(hw>>24)};
    uchar tb[4] = {(uchar)(tw), (uchar)(tw>>8), (uchar)(tw>>16), (uchar)(tw>>24)};
    for (int b = 0; b < 4; ++b) {
      if (hb[b] < tb[b]) return 1;
      if (hb[b] > tb[b]) return 0;
    }
  }
  return 1;
}

__kernel void search(
    __constant uint *header_words, // 23 u32 (92 bytes)
    __constant uint *target_words, // 8 u32
    const uint base_lo,
    const uint base_hi,
    const uint iters,
    __global uint *result // [0]=found, [1]=nonce_lo, [2]=nonce_hi, [3..10]=hash words
) {
  uint gid = get_global_id(0);
  uint w[23];
  for (int i = 0; i < 23; ++i) w[i] = header_words[i];

  // base + gid, then + threads each iter
  ulong base = ((ulong)base_hi << 32) | (ulong)base_lo;
  ulong threads = (ulong)get_global_size(0);
  ulong start = base + (ulong)gid;

  for (uint i = 0; i < iters; ++i) {
    if (atomic_or(&result[0], 0u) != 0u) return;
    ulong nonce = start + (ulong)i * threads;
    uint nlo = (uint)(nonce & 0xffffffffu);
    uint nhi = (uint)(nonce >> 32);
    uint h[8];
    blake3_92(w, nlo, nhi, h);
    if (meets_target(h, target_words)) {
      if (atomic_cmpxchg(&result[0], 0u, 1u) == 0u) {
        result[1] = nlo;
        result[2] = nhi;
        for (int k = 0; k < 8; ++k) result[3 + k] = h[k];
      }
      return;
    }
  }
}
