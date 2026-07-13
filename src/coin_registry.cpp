#include "core_types.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace miner {
namespace {

const CoinProfile kCoins[] = {
    {
        "alpha",
        "Alphanumeric (ALPHA)",
        AlgoId::Blake3An,
        ProtocolId::XmrStratum,
        TargetMode::BigEndianBytes,
        92,
        44,
        8,
        0x0000FFFFFFFFFFFFULL,
        "blake3-an",
        "Monero-style stratum; 92-byte header; BLAKE3 PoW",
    },
    {
        "lattica",
        "Lattica (LTA)",
        AlgoId::Sha3d,
        ProtocolId::BtcStratum,
        TargetMode::LittleEndianUint,
        80,
        76,
        4,
        0xFFFFFFFFULL,
        "sha3d",
        "Bitcoin-style stratum or solo GBT; 80-byte header; SHA3-256d PoW",
    },
};

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

}  // namespace

const CoinProfile* find_coin(const std::string& id) {
  const std::string key = lower(id);
  for (const auto& c : kCoins) {
    if (key == c.id) return &c;
  }
  // Aliases / algo shortcuts
  if (key == "lta" || key == "sha3d" || key == "sha3-256d") return &kCoins[1];
  if (key == "blake3-an" || key == "blake3an" || key == "blake3") return &kCoins[0];
  return nullptr;
}

std::vector<const CoinProfile*> list_coins() {
  return {&kCoins[0], &kCoins[1]};
}

const CoinProfile& default_coin() {
  // Prefer Lattica when this binary is used as a multi-coin miner; callers
  // should still pass --coin explicitly for clarity.
  return kCoins[1];
}

}  // namespace miner
