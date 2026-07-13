#include "core_types.hpp"
#include "cpu_miner.hpp"
#include "devfee.hpp"
#include "sha3.hpp"
#include "stratum.hpp"
#include "stratum_btc.hpp"
#include "util.hpp"

#ifdef ALPHA_HAS_OPENCL
#include "opencl_miner.hpp"
#endif
#ifdef ALPHA_HAS_HIP
#include "hip_miner.hpp"
#endif
#ifdef ALPHA_HAS_CUDA
#include "cuda_miner.hpp"
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> g_stop{false};
void on_sig(int) { g_stop = true; }

void usage(const char* argv0) {
  std::cout
      << "alpha-miner — modular multi-coin GPU/CPU pool miner\n"
      << "  Coins: alpha (blake3-an), lattica (sha3d)\n\n"
      << "Usage:\n"
      << "  " << argv0 << " -c COIN -o HOST:PORT -u WALLET.worker [options]\n\n"
      << "Options:\n"
      << "  -c, --coin NAME         alpha | lattica  (default: lattica)\n"
      << "  -a, --algo NAME         blake3-an | sha3d  (overrides coin default)\n"
      << "  -o, --url HOST:PORT     Pool stratum endpoint\n"
      << "  -u, --user USER         Wallet address[.worker]  (solo: prefix for some pools)\n"
      << "  -p, --pass PASS         Password (default: x)\n"
      << "  -b, --backend NAME      cpu | cuda | hip | opencl | auto (default: auto)\n"
      << "  -t, --threads N         CPU threads (default: hardware concurrency)\n"
      << "  -d, --devices LIST      GPU indices, e.g. 0,1,2\n"
      << "  -k, --kernel-dir PATH   Directory with *.cl kernels (default: kernels)\n"
      << "  -l, --list-devices      List CUDA/HIP/OpenCL GPUs and exit\n"
      << "      --list-coins        List built-in coins and exit\n"
      << "      --benchmark         Hashrate self-test for selected algo (CPU + optional GPU)\n"
      << "  -h, --help              Show help\n\n"
      << "Lattica examples (SHA3-256d, Bitcoin stratum):\n"
      << "  " << argv0 << " -c lattica -o stratum.example:3333 -u lta1q....rig1 -b cuda\n"
      << "  " << argv0 << " -c lattica -o eu.pool:3333 -u solo:lta1q.... -b opencl\n\n"
      << "ALPHA examples (blake3-an, Monero stratum):\n"
      << "  " << argv0 << " -c alpha -o eu.rplant.xyz:7176 -u YOUR_ALPHA.rig1 -b cuda\n\n"
      << "Adding a new coin: register CoinProfile in coin_registry.cpp and, if the\n"
      << "algo is new, add a kernel + backend path. Host/protocol loop is shared.\n";
}

bool parse_host_port(const std::string& url, std::string& host, uint16_t& port) {
  std::string u = url;
  auto strip = [&](const char* pfx) {
    if (u.rfind(pfx, 0) == 0) u = u.substr(std::strlen(pfx));
  };
  strip("stratum+tcp://");
  strip("stratum+tcps://");
  strip("stratum://");
  strip("tcp://");
  auto pos = u.rfind(':');
  if (pos == std::string::npos) return false;
  host = u.substr(0, pos);
  port = static_cast<uint16_t>(std::stoi(u.substr(pos + 1)));
  return !host.empty() && port > 0;
}

std::vector<int> parse_devices(const std::string& s) {
  std::vector<int> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    out.push_back(std::stoi(s.substr(i, j - i)));
    i = j + 1;
  }
  return out;
}

void print_coins() {
  std::cout << "Built-in coins:\n";
  for (auto* c : miner::list_coins()) {
    std::cout << "  " << c->id << " — " << c->display << "\n"
              << "      algo=" << c->algo_stratum_name
              << " protocol="
              << (c->protocol == miner::ProtocolId::XmrStratum
                      ? "xmr-stratum"
                      : c->protocol == miner::ProtocolId::BtcStratum ? "btc-stratum" : "gbt-rpc")
              << "\n"
              << "      " << c->notes << "\n";
  }
}

int run_benchmark(miner::AlgoId algo, const std::string& backend, int threads,
                  const std::vector<int>& /*devices*/) {
  using clock = std::chrono::steady_clock;
  const double secs = 3.0;
  if (algo == miner::AlgoId::Sha3d) {
    if (!miner::sha3_selftest()) {
      std::cerr << "[bench] SHA3 FIPS-202 self-test FAILED\n";
      return 1;
    }
    std::cout << "[bench] SHA3-256d FIPS-202 OK\n";
    uint8_t hdr[80] = {};
    for (int i = 0; i < 80; ++i) hdr[i] = (uint8_t)(i * 17 + 3);
    auto t0 = clock::now();
    uint64_t n = 0;
    uint8_t out[32];
    while (std::chrono::duration<double>(clock::now() - t0).count() < secs) {
      for (int k = 0; k < 256; ++k) {
        hdr[76] = (uint8_t)(n & 0xff);
        hdr[77] = (uint8_t)((n >> 8) & 0xff);
        hdr[78] = (uint8_t)((n >> 16) & 0xff);
        hdr[79] = (uint8_t)((n >> 24) & 0xff);
        miner::sha3_256d_header80(hdr, out);
        ++n;
      }
    }
    double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::cout << "[bench] CPU sha3d " << alpha::format_hashrate((double)n / dt)
              << "  (" << threads << " threads would scale ~linearly)\n";
  } else {
    uint8_t hdr[92] = {};
    for (int i = 0; i < 92; ++i) hdr[i] = (uint8_t)(i * 37 + 11);
    auto t0 = clock::now();
    uint64_t n = 0;
    uint8_t out[32];
    while (std::chrono::duration<double>(clock::now() - t0).count() < secs) {
      for (int k = 0; k < 256; ++k) {
        for (int i = 0; i < 8; ++i) hdr[44 + i] = (uint8_t)((n >> (8 * i)) & 0xff);
        alpha::blake3_header(hdr, out);
        ++n;
      }
    }
    double dt = std::chrono::duration<double>(clock::now() - t0).count();
    std::cout << "[bench] CPU blake3-an " << alpha::format_hashrate((double)n / dt) << "\n";
  }
  std::cout << "[bench] Use a full mining run with -b " << backend
            << " against a pool for GPU hashrate.\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string url, user, pass = "x", backend = "auto", kernel_dir = "kernels";
  std::string coin_id = "lattica";
  std::string algo_override;
  int threads = std::max(1u, std::thread::hardware_concurrency());
  std::vector<int> devices;
  bool list_only = false;
  bool list_coins = false;
  bool benchmark = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    } else if (a == "-c" || a == "--coin")
      coin_id = need(a.c_str());
    else if (a == "-a" || a == "--algo")
      algo_override = need(a.c_str());
    else if (a == "-o" || a == "--url")
      url = need(a.c_str());
    else if (a == "-u" || a == "--user")
      user = need(a.c_str());
    else if (a == "-p" || a == "--pass")
      pass = need(a.c_str());
    else if (a == "-b" || a == "--backend")
      backend = need(a.c_str());
    else if (a == "-t" || a == "--threads")
      threads = std::stoi(need(a.c_str()));
    else if (a == "-d" || a == "--devices")
      devices = parse_devices(need(a.c_str()));
    else if (a == "-k" || a == "--kernel" || a == "--kernel-dir")
      kernel_dir = need(a.c_str());
    else if (a == "-l" || a == "--list-devices")
      list_only = true;
    else if (a == "--list-coins")
      list_coins = true;
    else if (a == "--benchmark")
      benchmark = true;
    else {
      std::cerr << "unknown arg: " << a << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (list_coins) {
    print_coins();
    return 0;
  }

  if (list_only) {
    bool any = false;
#ifdef ALPHA_HAS_CUDA
    std::cout << "CUDA devices:\n";
    for (auto& s : alpha::CudaMiner::list_devices()) std::cout << "  " << s << "\n";
    any = true;
#endif
#ifdef ALPHA_HAS_HIP
    std::cout << "HIP devices:\n";
    for (auto& s : alpha::HipMiner::list_devices()) std::cout << "  " << s << "\n";
    any = true;
#endif
#ifdef ALPHA_HAS_OPENCL
    std::cout << "OpenCL devices:\n";
    for (auto& s : alpha::OpenClMiner::list_devices()) std::cout << "  " << s << "\n";
    any = true;
#endif
    if (!any) {
      std::cerr << "No GPU backends compiled into this binary\n";
      return 1;
    }
    return 0;
  }

  const miner::CoinProfile* coin = miner::find_coin(coin_id);
  if (!coin) {
    std::cerr << "unknown coin: " << coin_id << " (try --list-coins)\n";
    return 2;
  }
  miner::AlgoId algo = coin->algo;
  if (!algo_override.empty()) {
    if (algo_override == "sha3d" || algo_override == "sha3-256d")
      algo = miner::AlgoId::Sha3d;
    else if (algo_override == "blake3-an" || algo_override == "blake3")
      algo = miner::AlgoId::Blake3An;
    else {
      std::cerr << "unknown algo: " << algo_override << "\n";
      return 2;
    }
  }

  if (benchmark) return run_benchmark(algo, backend, threads, devices);

  if (url.empty() || user.empty()) {
    usage(argv[0]);
    return 2;
  }
  std::string host;
  uint16_t port = 0;
  if (!parse_host_port(url, host, port)) {
    std::cerr << "bad -o URL, expected host:port\n";
    return 2;
  }

  std::signal(SIGINT, on_sig);
  std::signal(SIGTERM, on_sig);

  // Algo self-check
  if (algo == miner::AlgoId::Sha3d) {
    if (!miner::sha3_selftest()) {
      std::cerr << "[self-check] SHA3-256 FIPS-202 FAILED — refuse to mine\n";
      return 1;
    }
    uint8_t hdr[80] = {};
    for (int i = 0; i < 80; ++i) hdr[i] = (uint8_t)(i * 37 + 11);
    uint8_t out[32];
    miner::sha3_256d_header80(hdr, out);
    std::cout << "[self-check] sha3d=" << alpha::to_hex(out, 8) << "... OK\n";
  } else {
    uint8_t hdr[92];
    for (int i = 0; i < 92; ++i) hdr[i] = (uint8_t)(i * 37 + 11);
    uint64_t nonce = 0x0123456789ABCDEFULL;
    for (int i = 0; i < 8; ++i) hdr[44 + i] = (uint8_t)((nonce >> (8 * i)) & 0xff);
    uint8_t out[32];
    alpha::blake3_header(hdr, out);
    std::cout << "[self-check] blake3=" << alpha::to_hex(out, 8) << "...\n";
  }

  std::cout << "[main] coin=" << coin->id << " (" << coin->display << ") algo="
            << (algo == miner::AlgoId::Sha3d ? "sha3d" : "blake3-an") << "\n";

  miner::JobMux jobs;
  // Devfee only for ALPHA (configured wallet). Lattica: no built-in fee by default.
  const bool alpha_fee = (coin->algo == miner::AlgoId::Blake3An) && alpha::devfee::enabled();
  jobs.fee_percent = alpha_fee ? alpha::devfee::kPercent : 0;
  jobs.fee_enabled = alpha_fee;

  std::unique_ptr<miner::IWorkSource> work;
  std::unique_ptr<miner::IWorkSource> fee_work;
  std::unique_ptr<miner::IShareSink> sink;

  if (coin->protocol == miner::ProtocolId::BtcStratum || algo == miner::AlgoId::Sha3d) {
    auto btc = std::make_unique<miner::BtcStratumClient>(host, port, user, pass,
                                                         "alpha-miner/0.3.0-lattica", algo);
    work = std::move(btc);
    sink = std::make_unique<alpha::WorkShareRouter>(*work, nullptr);
  } else {
    auto xmr = std::make_unique<alpha::StratumClient>(host, port, user, pass, "alpha-miner/0.3.0",
                                                      false);
    if (alpha_fee) {
      std::string fee_user = std::string(alpha::devfee::kAddress) + "." + alpha::devfee::kWorker;
      auto fee = std::make_unique<alpha::StratumClient>(host, port, fee_user, pass,
                                                        "alpha-miner/0.3.0-fee", true);
      fee_work = std::move(fee);
      std::cout << "[devfee] " << alpha::devfee::kPercent << "% enabled → "
                << alpha::devfee::kAddress << "\n";
    } else {
      std::cout << "[devfee] disabled\n";
    }
    // ShareRouter needs StratumClient& — keep typed pointers for ALPHA path
    // Rebuild: store raw and wrap
    work = std::move(xmr);
    sink = std::make_unique<alpha::WorkShareRouter>(*work, fee_work.get());
  }

  std::unique_ptr<alpha::CpuMiner> cpu;
#ifdef ALPHA_HAS_OPENCL
  std::unique_ptr<alpha::OpenClMiner> ocl;
#endif
#ifdef ALPHA_HAS_HIP
  std::unique_ptr<alpha::HipMiner> hip;
#endif
#ifdef ALPHA_HAS_CUDA
  std::unique_ptr<alpha::CudaMiner> cuda;
#endif

  bool use_gpu = false;
  std::string chosen;

  // auto: CUDA → HIP (blake3 only) → OpenCL → CPU
  if (backend == "cuda" || backend == "auto") {
#ifdef ALPHA_HAS_CUDA
    cuda = std::make_unique<alpha::CudaMiner>(jobs, *sink, devices, algo);
    if (cuda->init()) {
      use_gpu = true;
      chosen = "cuda";
    } else if (backend == "cuda") {
      std::cerr << "[main] CUDA requested but unavailable\n";
      return 1;
    } else {
      cuda.reset();
    }
#else
    if (backend == "cuda") {
      std::cerr << "[main] binary built without CUDA\n";
      return 1;
    }
#endif
  }

  if (!use_gpu && (backend == "hip" || backend == "auto") && algo == miner::AlgoId::Blake3An) {
#ifdef ALPHA_HAS_HIP
    hip = std::make_unique<alpha::HipMiner>(jobs, *sink, devices, algo);
    if (hip->init()) {
      use_gpu = true;
      chosen = "hip";
    } else if (backend == "hip") {
      std::cerr << "[main] HIP requested but unavailable\n";
      return 1;
    } else {
      hip.reset();
    }
#else
    if (backend == "hip") {
      std::cerr << "[main] binary built without HIP\n";
      return 1;
    }
#endif
  }

  if (!use_gpu && (backend == "opencl" || backend == "auto")) {
#ifdef ALPHA_HAS_OPENCL
    ocl = std::make_unique<alpha::OpenClMiner>(jobs, *sink, kernel_dir, devices, algo);
    if (ocl->init()) {
      use_gpu = true;
      chosen = "opencl";
    } else if (backend == "opencl") {
      std::cerr << "[main] OpenCL requested but unavailable\n";
      return 1;
    } else {
      ocl.reset();
    }
#else
    if (backend == "opencl") {
      std::cerr << "[main] binary built without OpenCL\n";
      return 1;
    }
#endif
  }

  if (!use_gpu) {
    cpu = std::make_unique<alpha::CpuMiner>(jobs, *sink, threads);
    chosen = "cpu threads=" + std::to_string(threads);
  }

  std::cout << "[main] backend=" << chosen << "\n";
  work->start(jobs.user);
  if (fee_work) fee_work->start(jobs.fee);
#ifdef ALPHA_HAS_CUDA
  if (cuda) cuda->start();
#endif
#ifdef ALPHA_HAS_HIP
  if (hip) hip->start();
#endif
#ifdef ALPHA_HAS_OPENCL
  if (ocl) ocl->start();
#endif
  if (cpu) cpu->start();

  while (!g_stop) {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    double hr = 0;
    if (cpu) hr += cpu->hashrate();
#ifdef ALPHA_HAS_OPENCL
    if (ocl) hr += ocl->hashrate();
#endif
#ifdef ALPHA_HAS_HIP
    if (hip) hr += hip->hashrate();
#endif
#ifdef ALPHA_HAS_CUDA
    if (cuda) hr += cuda->hashrate();
#endif
    std::cout << "[stats] " << alpha::format_hashrate(hr)
              << " connected=" << (work->connected() ? "yes" : "no") << std::endl;
  }

  if (cpu) cpu->stop();
#ifdef ALPHA_HAS_OPENCL
  if (ocl) ocl->stop();
#endif
#ifdef ALPHA_HAS_HIP
  if (hip) hip->stop();
#endif
#ifdef ALPHA_HAS_CUDA
  if (cuda) cuda->stop();
#endif
  work->stop();
  if (fee_work) fee_work->stop();
  std::cout << "[main] stopped\n";
  return 0;
}
