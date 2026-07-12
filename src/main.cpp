#include "cpu_miner.hpp"
#include "devfee.hpp"
#include "job.hpp"
#include "stratum.hpp"
#include "util.hpp"

#ifdef ALPHA_HAS_OPENCL
#include "opencl_miner.hpp"
#endif
#ifdef ALPHA_HAS_HIP
#include "hip_miner.hpp"
#endif

#include <atomic>
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
      << "alpha-miner — Alphanumeric (blake3-an) GPU/CPU pool miner\n\n"
      << "Usage:\n"
      << "  " << argv0 << " -o HOST:PORT -u WALLET.worker [options]\n\n"
      << "Options:\n"
      << "  -o, --url HOST:PORT     Pool stratum (e.g. eu.rplant.xyz:7176)\n"
      << "  -u, --user USER         Wallet address[.worker]\n"
      << "  -p, --pass PASS         Password (default: x)\n"
      << "  -b, --backend NAME      cpu | hip | opencl | auto (default: auto)\n"
      << "  -t, --threads N         CPU threads (default: hardware concurrency)\n"
      << "  -d, --devices LIST      GPU device indices, e.g. 0,1\n"
      << "  -k, --kernel PATH       Path to blake3_an.cl (OpenCL only)\n"
      << "  -l, --list-devices      List HIP/OpenCL GPUs and exit\n"
      << "  -h, --help              Show help\n\n"
      << "Examples:\n"
      << "  # AMD via HIP (ROCm) — preferred on Radeon\n"
      << "  " << argv0 << " -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b hip\n\n"
      << "  # NVIDIA or AMD via OpenCL\n"
      << "  " << argv0 << " -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b opencl\n\n"
      << "  # CPU only\n"
      << "  " << argv0 << " -o eu.rplant.xyz:7176 -u YOUR_ALPHA_ADDRESS.rig1 -b cpu -t 16\n\n"
      << "Pool protocol: monero-style login/job/submit, algo blake3-an, 92-byte header.\n"
      << "Reference pool: stratum+tcp://eu.rplant.xyz:7176 (port 7176).\n";
}

bool parse_host_port(const std::string& url, std::string& host, uint16_t& port) {
  std::string u = url;
  auto strip = [&](const char* pfx) {
    if (u.rfind(pfx, 0) == 0) u = u.substr(std::strlen(pfx));
  };
  strip("stratum+tcp://");
  strip("stratum+tcps://");
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
}  // namespace

int main(int argc, char** argv) {
  std::string url, user, pass = "x", backend = "auto", kernel = "kernels/blake3_an.cl";
  int threads = std::max(1u, std::thread::hardware_concurrency());
  std::vector<int> devices;
  bool list_only = false;

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
    } else if (a == "-o" || a == "--url")
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
    else if (a == "-k" || a == "--kernel")
      kernel = need(a.c_str());
    else if (a == "-l" || a == "--list-devices")
      list_only = true;
    else {
      std::cerr << "unknown arg: " << a << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (list_only) {
    bool any = false;
#ifdef ALPHA_HAS_HIP
    std::cout << "HIP devices:\n";
    auto hip = alpha::HipMiner::list_devices();
    if (hip.empty()) std::cout << "  (none)\n";
    for (auto& s : hip) std::cout << "  " << s << "\n";
    any = true;
#endif
#ifdef ALPHA_HAS_OPENCL
    std::cout << "OpenCL devices:\n";
    auto ocl = alpha::OpenClMiner::list_devices();
    if (ocl.empty()) std::cout << "  (none)\n";
    for (auto& s : ocl) std::cout << "  " << s << "\n";
    any = true;
#endif
    if (!any) {
      std::cerr << "No GPU backends compiled into this binary\n";
      return 1;
    }
    return 0;
  }

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

  // self-check BLAKE3
  {
    uint8_t hdr[92];
    for (int i = 0; i < 92; ++i) hdr[i] = (uint8_t)(i * 37 + 11);
    uint64_t nonce = 0x0123456789ABCDEFULL;
    for (int i = 0; i < 8; ++i) hdr[44 + i] = (uint8_t)((nonce >> (8 * i)) & 0xff);
    uint8_t out[32];
    alpha::blake3_header(hdr, out);
    std::cout << "[self-check] blake3=" << alpha::to_hex(out, 8) << "...\n";
  }

  alpha::JobMux jobs;
  jobs.fee_percent = alpha::devfee::kPercent;
  jobs.fee_enabled = alpha::devfee::enabled();

  // User login = wallet[.worker] as provided.
  alpha::StratumClient stratum(host, port, user, pass, "alpha-miner/0.2.0", false);

  // Optional 2% developer fee: second pool session to fee wallet.
  std::unique_ptr<alpha::StratumClient> fee_stratum;
  if (jobs.fee_enabled) {
    std::string fee_user = std::string(alpha::devfee::kAddress) + "." + alpha::devfee::kWorker;
    fee_stratum = std::make_unique<alpha::StratumClient>(host, port, fee_user, pass,
                                                         "alpha-miner/0.2.0-fee", true);
    std::cout << "[devfee] " << alpha::devfee::kPercent
              << "% enabled → " << alpha::devfee::kAddress << "\n";
  } else {
    std::cout << "[devfee] disabled (set ALPHA_DEV_FEE_ADDRESS at build time)\n";
  }

  alpha::ShareRouter router(stratum, fee_stratum.get());

  std::unique_ptr<alpha::CpuMiner> cpu;
#ifdef ALPHA_HAS_OPENCL
  std::unique_ptr<alpha::OpenClMiner> ocl;
#endif
#ifdef ALPHA_HAS_HIP
  std::unique_ptr<alpha::HipMiner> hip;
#endif

  bool use_gpu = false;
  std::string chosen;

  // Prefer HIP on AMD when auto. Probe devices before connecting to the pool.
  if (backend == "hip" || backend == "auto") {
#ifdef ALPHA_HAS_HIP
    hip = std::make_unique<alpha::HipMiner>(jobs, router, devices);
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
      std::cerr << "[main] binary built without HIP (need hipcc / ROCm)\n";
      return 1;
    }
#endif
  }

  if (!use_gpu && (backend == "opencl" || backend == "auto")) {
#ifdef ALPHA_HAS_OPENCL
    ocl = std::make_unique<alpha::OpenClMiner>(jobs, router, kernel, devices);
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
    cpu = std::make_unique<alpha::CpuMiner>(jobs, router, threads);
    chosen = "cpu threads=" + std::to_string(threads);
  }

  std::cout << "[main] backend=" << chosen << "\n";
  stratum.start(jobs.user);
  if (fee_stratum) fee_stratum->start(jobs.fee);
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
    std::cout << "[stats] " << alpha::format_hashrate(hr)
              << " connected=" << (stratum.connected() ? "yes" : "no");
    if (fee_stratum)
      std::cout << " fee=" << (fee_stratum->connected() ? "yes" : "no");
    std::cout << std::endl;
  }

  if (cpu) cpu->stop();
#ifdef ALPHA_HAS_OPENCL
  if (ocl) ocl->stop();
#endif
#ifdef ALPHA_HAS_HIP
  if (hip) hip->stop();
#endif
  stratum.stop();
  if (fee_stratum) fee_stratum->stop();
  std::cout << "[main] stopped\n";
  return 0;
}
