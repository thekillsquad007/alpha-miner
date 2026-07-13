#ifdef ALPHA_HAS_OPENCL
#include "opencl_miner.hpp"
#include "util.hpp"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

namespace alpha {
namespace {

std::string load_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open kernel: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void check(cl_int err, const char* what) {
  if (err != CL_SUCCESS) {
    throw std::runtime_error(std::string(what) + " err=" + std::to_string(err));
  }
}

std::string join_path(const std::string& dir, const std::string& file) {
  if (dir.empty()) return file;
  if (dir.back() == '/' || dir.back() == '\\') return dir + file;
  return dir + "/" + file;
}

}  // namespace

std::vector<std::string> OpenClMiner::list_devices() {
  std::vector<std::string> out;
  cl_uint np = 0;
  clGetPlatformIDs(0, nullptr, &np);
  std::vector<cl_platform_id> plats(np);
  if (np) clGetPlatformIDs(np, plats.data(), nullptr);
  int idx = 0;
  for (auto p : plats) {
    cl_uint nd = 0;
    clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
    std::vector<cl_device_id> devs(nd);
    if (nd) clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
    for (auto d : devs) {
      char name[256] = {};
      clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, nullptr);
      out.push_back(std::to_string(idx++) + ": " + name);
    }
  }
  return out;
}

OpenClMiner::OpenClMiner(miner::JobMux& jobs, miner::IShareSink& sink, std::string kernel_dir,
                         const std::vector<int>& devices, miner::AlgoId algo)
    : jobs_(jobs),
      sink_(sink),
      kernel_dir_(std::move(kernel_dir)),
      devices_(devices),
      algo_(algo) {}

OpenClMiner::~OpenClMiner() { stop(); }

bool OpenClMiner::init() {
  auto list = list_devices();
  if (list.empty()) {
    std::cerr << "[opencl] no GPU devices found\n";
    return false;
  }
  std::cout << "[opencl] devices:\n";
  for (auto& s : list) std::cout << "  " << s << "\n";
  return true;
}

void OpenClMiner::start() {
  stop_ = false;
  hashes_ = 0;
  start_ms_ = now_ms();
  auto all = list_devices();
  std::vector<int> use = devices_;
  if (use.empty()) {
    for (int i = 0; i < (int)all.size(); ++i) use.push_back(i);
  }
  for (int i = 0; i < (int)use.size(); ++i) {
    workers_.emplace_back([this, dev = use[i], i] { worker(dev, i); });
  }
}

void OpenClMiner::stop() {
  stop_ = true;
  for (auto& t : workers_)
    if (t.joinable()) t.join();
  workers_.clear();
}

double OpenClMiner::hashrate() const {
  uint64_t ms = now_ms() - start_ms_;
  if (!ms) return 0;
  return (double)hashes_.load() * 1000.0 / (double)ms;
}

void OpenClMiner::worker(int device_index, int logical_id) {
  if (algo_ == miner::AlgoId::Sha3d)
    worker_sha3d(device_index, logical_id);
  else
    worker_blake3(device_index, logical_id);
}

void OpenClMiner::worker_blake3(int device_index, int logical_id) {
  try {
    cl_uint np = 0;
    clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    std::vector<cl_device_id> all;
    for (auto p : plats) {
      cl_uint nd = 0;
      clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
      size_t old = all.size();
      all.resize(old + nd);
      if (nd) clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, all.data() + old, nullptr);
    }
    if (device_index < 0 || device_index >= (int)all.size()) {
      std::cerr << "[opencl] bad device index " << device_index << "\n";
      return;
    }
    cl_device_id dev = all[device_index];
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    check(err, "clCreateCommandQueue");
    std::string src = load_file(join_path(kernel_dir_, "blake3_an.cl"));
    const char* srcp = src.c_str();
    size_t srcl = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcp, &srcl, &err);
    check(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t log_sz = 0;
      clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
      std::string log(log_sz, '\0');
      clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
      std::cerr << "[opencl] build log:\n" << log << "\n";
      throw std::runtime_error("clBuildProgram failed");
    }
    cl_kernel kern = clCreateKernel(prog, "search", &err);
    check(err, "clCreateKernel");

    const size_t GLOBAL = 256 * 1024;
    const uint32_t ITERS = 64;
    cl_mem header_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 23 * 4, nullptr, &err);
    cl_mem target_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 8 * 4, nullptr, &err);
    cl_mem result_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 11 * 4, nullptr, &err);
    check(err, "buffers");

    char name[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cout << "[opencl/blake3] mining on device " << device_index << " (" << name << ")\n";

    while (!stop_) {
      miner::Job job = jobs_.get();
      if (job.job_id.empty() || job.epoch == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      const uint64_t epoch = job.epoch;
      const bool fee = job.is_devfee;
      uint32_t words[23] = {};
      for (int i = 0; i < 23; ++i) {
        words[i] = (uint32_t)job.header[i * 4] | ((uint32_t)job.header[i * 4 + 1] << 8) |
                   ((uint32_t)job.header[i * 4 + 2] << 16) | ((uint32_t)job.header[i * 4 + 3] << 24);
      }
      words[11] = 0;
      words[12] = 0;
      uint32_t twords[8] = {};
      for (int i = 0; i < 8; ++i) {
        twords[i] = (uint32_t)job.target[i * 4] | ((uint32_t)job.target[i * 4 + 1] << 8) |
                    ((uint32_t)job.target[i * 4 + 2] << 16) | ((uint32_t)job.target[i * 4 + 3] << 24);
      }
      clEnqueueWriteBuffer(q, header_buf, CL_TRUE, 0, sizeof(words), words, 0, nullptr, nullptr);
      clEnqueueWriteBuffer(q, target_buf, CL_TRUE, 0, sizeof(twords), twords, 0, nullptr, nullptr);

      uint64_t cursor = job.nonce_fixed | ((uint64_t)logical_id << 40);
      while (!stop_) {
        miner::Job latest = jobs_.get();
        if (latest.job_id != job.job_id || latest.epoch != epoch || latest.is_devfee != fee) break;

        uint32_t zero[11] = {};
        clEnqueueWriteBuffer(q, result_buf, CL_TRUE, 0, sizeof(zero), zero, 0, nullptr, nullptr);
        uint32_t base_lo = (uint32_t)(cursor & 0xffffffffu);
        uint32_t base_hi = (uint32_t)(cursor >> 32);
        uint32_t iters = ITERS;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &header_buf);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &target_buf);
        clSetKernelArg(kern, 2, sizeof(uint32_t), &base_lo);
        clSetKernelArg(kern, 3, sizeof(uint32_t), &base_hi);
        clSetKernelArg(kern, 4, sizeof(uint32_t), &iters);
        clSetKernelArg(kern, 5, sizeof(cl_mem), &result_buf);
        size_t g = GLOBAL;
        check(clEnqueueNDRangeKernel(q, kern, 1, nullptr, &g, nullptr, 0, nullptr, nullptr),
              "enqueue");
        clFinish(q);
        uint32_t res[11] = {};
        clEnqueueReadBuffer(q, result_buf, CL_TRUE, 0, sizeof(res), res, 0, nullptr, nullptr);
        hashes_.fetch_add((uint64_t)GLOBAL * ITERS, std::memory_order_relaxed);
        if (res[0]) {
          uint64_t nonce = ((uint64_t)res[2] << 32) | res[1];
          uint8_t header[92];
          std::memcpy(header, job.header.data(), 92);
          for (int i = 0; i < 8; ++i) header[44 + i] = (uint8_t)((nonce >> (8 * i)) & 0xff);
          uint8_t hash[32];
          blake3_header(header, hash);
          if (hash_meets_job_target(hash, job)) {
            miner::Share s;
            s.job_id = job.job_id;
            s.nonce = nonce;
            std::memcpy(s.hash.data(), hash, 32);
            s.job_epoch = epoch;
            s.is_devfee = fee;
            std::cout << (fee ? "[opencl/fee] " : "[opencl] ")
                      << "share nonce=" << to_hex(header + 44, 8) << std::endl;
            sink_.submit(s);
          }
        }
        cursor += (uint64_t)GLOBAL * ITERS;
        cursor = (cursor & 0x0000FFFFFFFFFFFFULL) | job.nonce_fixed;
      }
    }

    clReleaseMemObject(header_buf);
    clReleaseMemObject(target_buf);
    clReleaseMemObject(result_buf);
    clReleaseKernel(kern);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
  } catch (const std::exception& e) {
    std::cerr << "[opencl/blake3] worker error: " << e.what() << "\n";
  }
}

void OpenClMiner::worker_sha3d(int device_index, int logical_id) {
  try {
    cl_uint np = 0;
    clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    std::vector<cl_device_id> all;
    for (auto p : plats) {
      cl_uint nd = 0;
      clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
      size_t old = all.size();
      all.resize(old + nd);
      if (nd) clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, all.data() + old, nullptr);
    }
    if (device_index < 0 || device_index >= (int)all.size()) {
      std::cerr << "[opencl] bad device index " << device_index << "\n";
      return;
    }
    cl_device_id dev = all[device_index];
    cl_int err = 0;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    check(err, "clCreateCommandQueue");

    std::string kern_path = join_path(kernel_dir_, "sha3d.cl");
    std::string src = load_file(kern_path);
    const char* srcp = src.c_str();
    size_t srcl = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &srcp, &srcl, &err);
    check(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL1.2 -cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t log_sz = 0;
      clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
      std::string log(log_sz, '\0');
      clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
      std::cerr << "[opencl/sha3d] build log:\n" << log << "\n";
      throw std::runtime_error("clBuildProgram failed");
    }
    cl_kernel kern = clCreateKernel(prog, "search_sha3d", &err);
    check(err, "clCreateKernel search_sha3d");

    // Match Lattica official miner: 1M global, 256 local, 256 iters for high occupancy.
    const size_t GLOBAL = size_t{1} << 20;
    const size_t LOCAL = 256;
    const uint32_t ITERS = 256;

    cl_mem hdr_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 10 * sizeof(cl_ulong), nullptr, &err);
    cl_mem tgt_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 4 * sizeof(cl_ulong), nullptr, &err);
    cl_mem found_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    check(err, "buffers");

    char name[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cout << "[opencl/sha3d] mining on device " << device_index << " (" << name << ")\n";

    while (!stop_) {
      miner::Job job = jobs_.get();
      if (job.job_id.empty() || job.epoch == 0 || job.header_len != 80) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      const uint64_t epoch = job.epoch;
      const bool fee = job.is_devfee;

      cl_ulong hdr[10] = {};
      for (int i = 0; i < 10; ++i) {
        hdr[i] = 0;
        for (int b = 0; b < 8; ++b) hdr[i] |= (cl_ulong)job.header[i * 8 + b] << (8 * b);
      }
      cl_ulong tgt[4] = {};
      for (int i = 0; i < 4; ++i) {
        for (int b = 0; b < 8; ++b) tgt[i] |= (cl_ulong)job.target[i * 8 + b] << (8 * b);
      }
      clEnqueueWriteBuffer(q, hdr_buf, CL_TRUE, 0, sizeof(hdr), hdr, 0, nullptr, nullptr);
      clEnqueueWriteBuffer(q, tgt_buf, CL_TRUE, 0, sizeof(tgt), tgt, 0, nullptr, nullptr);

      // Partition 32-bit nonce space across devices.
      uint64_t cursor = ((uint64_t)logical_id * 0x10000000ULL) & 0xFFFFFFFFull;
      const uint64_t stride = (uint64_t)GLOBAL * ITERS;

      while (!stop_) {
        miner::Job latest = jobs_.get();
        if (latest.job_id != job.job_id || latest.epoch != epoch || latest.is_devfee != fee) break;

        cl_uint found = 0xFFFFFFFFu;
        clEnqueueWriteBuffer(q, found_buf, CL_TRUE, 0, sizeof(found), &found, 0, nullptr, nullptr);
        cl_ulong base = cursor;
        cl_uint iters = ITERS;
        clSetKernelArg(kern, 0, sizeof(cl_mem), &hdr_buf);
        clSetKernelArg(kern, 1, sizeof(cl_mem), &tgt_buf);
        clSetKernelArg(kern, 2, sizeof(cl_ulong), &base);
        clSetKernelArg(kern, 3, sizeof(cl_uint), &iters);
        clSetKernelArg(kern, 4, sizeof(cl_mem), &found_buf);
        size_t g = GLOBAL, l = LOCAL;
        check(clEnqueueNDRangeKernel(q, kern, 1, nullptr, &g, &l, 0, nullptr, nullptr), "enqueue");
        clFinish(q);
        clEnqueueReadBuffer(q, found_buf, CL_TRUE, 0, sizeof(found), &found, 0, nullptr, nullptr);
        hashes_.fetch_add(stride, std::memory_order_relaxed);

        if (found != 0xFFFFFFFFu) {
          uint8_t header[80];
          std::memcpy(header, job.header.data(), 80);
          header[76] = (uint8_t)(found & 0xff);
          header[77] = (uint8_t)((found >> 8) & 0xff);
          header[78] = (uint8_t)((found >> 16) & 0xff);
          header[79] = (uint8_t)((found >> 24) & 0xff);
          uint8_t hash[32];
          hash_job_header(job, header, hash);
          if (hash_meets_job_target(hash, job)) {
            miner::Share s;
            s.job_id = job.job_id;
            s.nonce = found;
            std::memcpy(s.hash.data(), hash, 32);
            s.job_epoch = epoch;
            s.is_devfee = fee;
            s.extranonce2_hex = job.extranonce2_hex;
            s.ntime_hex = job.ntime_hex;
            std::cout << (fee ? "[opencl/sha3d/fee] " : "[opencl/sha3d] ")
                      << "share nonce=0x" << std::hex << found << std::dec
                      << " hash=" << to_hex(hash, 8) << "...\n";
            sink_.submit(s);
          }
        }
        cursor = (cursor + stride) & 0xFFFFFFFFull;
      }
    }

    clReleaseMemObject(hdr_buf);
    clReleaseMemObject(tgt_buf);
    clReleaseMemObject(found_buf);
    clReleaseKernel(kern);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
  } catch (const std::exception& e) {
    std::cerr << "[opencl/sha3d] worker error: " << e.what() << "\n";
  }
}

}  // namespace alpha
#endif
