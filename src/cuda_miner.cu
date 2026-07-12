// Optional CUDA backend for NVIDIA. Build with -DALPHA_MINER_CUDA=ON.
// Mirrors the OpenCL search loop; uses the same 92-byte BLAKE3 header layout.

#ifdef ALPHA_HAS_CUDA
#include "job.hpp"
#include "stratum.hpp"
#include "util.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

// For brevity the CUDA path falls back to launching many CPU-verified batches
// only when a full device kernel is not compiled in this skeleton. Prefer the
// OpenCL binary on mixed fleets; enable this file's full kernel for pure NVIDIA.

namespace alpha {

// Placeholder export so the binary links when CUDA is enabled.
void cuda_miner_note() {
  std::cerr << "[cuda] Full CUDA kernel not compiled in this build; use OpenCL backend "
               "(-b opencl) or gpuminer-rplant-cuda for production NVIDIA.\n";
}

}  // namespace alpha
#endif
