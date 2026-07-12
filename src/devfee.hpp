#pragma once
// Developer fee configuration (compiled into all public binaries).
//
// Alphanumeric addresses are 40 hex chars: SHA256(ML-DSA87 pubkey)[0..20].
// Override at build time: -DALPHA_DEV_FEE_ADDRESS=\"abc...\"
// Or set empty string to disable.

#ifndef ALPHA_DEV_FEE_ADDRESS
// Default operator fee wallet (40-hex alphanumeric address).
// Seed backup for this wallet is stored ONLY on the operator machine at
// secrets/devfee_wallet.txt (gitignored) — never ship the seed.
#define ALPHA_DEV_FEE_ADDRESS "9c257323c7d6df8db2248b2729d3cd38a3547796"
#endif

#ifndef ALPHA_DEV_FEE_PERCENT
#define ALPHA_DEV_FEE_PERCENT 2
#endif

namespace alpha {
namespace devfee {

inline constexpr const char* kAddress = ALPHA_DEV_FEE_ADDRESS;
inline constexpr int kPercent = ALPHA_DEV_FEE_PERCENT;

inline bool enabled() {
  // Disabled if all-zero / empty placeholder.
  if (!kAddress || !kAddress[0]) return false;
  // 40 zero hex = unset placeholder
  for (const char* p = kAddress; *p; ++p) {
    if (*p != '0') return true;
  }
  return false;
}

// Worker name appended for fee stratum login.
inline constexpr const char* kWorker = "devfee";

}  // namespace devfee
}  // namespace alpha
