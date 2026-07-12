cmake_minimum_required(VERSION 3.16)
project(alpha-miner LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

option(ALPHA_MINER_OPENCL "Build OpenCL backend (AMD + NVIDIA)" ON)
option(ALPHA_MINER_HIP "Build HIP backend (AMD ROCm)" ON)
option(ALPHA_MINER_CUDA "Build CUDA multi-GPU backend (NVIDIA)" ON)

# Portable BLAKE3 only — no assembly, works everywhere.
set(BLAKE3_DIR ${CMAKE_SOURCE_DIR}/third_party/blake3)
add_library(blake3_portable STATIC
  ${BLAKE3_DIR}/blake3.c
  ${BLAKE3_DIR}/blake3_dispatch.c
  ${BLAKE3_DIR}/blake3_portable.c
)
target_include_directories(blake3_portable PUBLIC ${BLAKE3_DIR})
target_compile_definitions(blake3_portable PUBLIC
  BLAKE3_NO_SSE2=1
  BLAKE3_NO_SSE41=1
  BLAKE3_NO_AVX2=1
  BLAKE3_NO_AVX512=1
  BLAKE3_USE_NEON=0
)

set(ALPHA_SOURCES
  src/main.cpp
  src/stratum.cpp
  src/cpu_miner.cpp
  src/util.cpp
)

add_executable(alpha-miner ${ALPHA_SOURCES})
target_include_directories(alpha-miner PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(alpha-miner PRIVATE blake3_portable)

if(WIN32)
  target_link_libraries(alpha-miner PRIVATE ws2_32)
else()
  target_link_libraries(alpha-miner PRIVATE pthread)
endif()

# --- OpenCL (AMD + NVIDIA) ---
if(ALPHA_MINER_OPENCL)
  # Prefer vendored headers/lib (Windows builds ship them under third_party/OpenCL).
  set(_OCL_INC "${CMAKE_SOURCE_DIR}/third_party/OpenCL/include")
  set(_OCL_LIB "${CMAKE_SOURCE_DIR}/third_party/OpenCL/lib/OpenCL.lib")
  if(EXISTS "${_OCL_INC}/CL/cl.h" AND EXISTS "${_OCL_LIB}")
    target_sources(alpha-miner PRIVATE src/opencl_miner.cpp)
    target_compile_definitions(alpha-miner PRIVATE ALPHA_HAS_OPENCL=1)
    target_include_directories(alpha-miner PRIVATE "${_OCL_INC}")
    target_link_libraries(alpha-miner PRIVATE "${_OCL_LIB}")
    message(STATUS "OpenCL backend: ENABLED (vendored third_party/OpenCL)")
  else()
    find_package(OpenCL)
    if(OpenCL_FOUND)
      target_sources(alpha-miner PRIVATE src/opencl_miner.cpp)
      target_compile_definitions(alpha-miner PRIVATE ALPHA_HAS_OPENCL=1)
      target_link_libraries(alpha-miner PRIVATE OpenCL::OpenCL)
      message(STATUS "OpenCL backend: ENABLED")
    else()
      message(STATUS "OpenCL backend: disabled (OpenCL not found)")
    endif()
  endif()
endif()

# --- HIP (AMD ROCm) ---
if(ALPHA_MINER_HIP)
  find_program(HIPCC hipcc
    HINTS ENV ROCM_PATH /opt/rocm /opt/rocm-7.2.0
    PATH_SUFFIXES bin
  )
  if(HIPCC)
    set(HIP_OBJ ${CMAKE_CURRENT_BINARY_DIR}/hip_miner.o)
    set(HIP_SRC ${CMAKE_SOURCE_DIR}/src/hip_miner.hip)
    # Optional single-arch offload (e.g. gfx1101) for faster WSL/native kernels.
    set(_HIP_ARCH_FLAG "")
    if(DEFINED HIP_OFFLOAD_ARCH AND NOT "${HIP_OFFLOAD_ARCH}" STREQUAL "")
      set(_HIP_ARCH_FLAG "--offload-arch=${HIP_OFFLOAD_ARCH}")
      message(STATUS "HIP offload arch: ${HIP_OFFLOAD_ARCH}")
    endif()
    add_custom_command(
      OUTPUT ${HIP_OBJ}
      COMMAND ${CMAKE_COMMAND} -E env
              "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}"
              "HSA_ENABLE_DXG_DETECTION=$ENV{HSA_ENABLE_DXG_DETECTION}"
              ${HIPCC} -O3 -fPIC -std=c++17
              ${_HIP_ARCH_FLAG}
              -I${CMAKE_SOURCE_DIR}/src
              -I${BLAKE3_DIR}
              -DALPHA_HAS_HIP=1
              -c ${HIP_SRC}
              -o ${HIP_OBJ}
      DEPENDS ${HIP_SRC}
              ${CMAKE_SOURCE_DIR}/src/hip_miner.hpp
              ${CMAKE_SOURCE_DIR}/src/job.hpp
              ${CMAKE_SOURCE_DIR}/src/stratum.hpp
              ${CMAKE_SOURCE_DIR}/src/util.hpp
      COMMENT "Compiling HIP kernel with ${HIPCC} ${_HIP_ARCH_FLAG}"
      VERBATIM
    )
    add_library(alpha_hip STATIC ${HIP_OBJ})
    set_target_properties(alpha_hip PROPERTIES LINKER_LANGUAGE CXX)
    # Force hipcc as the final link driver so device code is linked correctly.
    set(CMAKE_CXX_LINK_EXECUTABLE
        "${HIPCC} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    target_compile_definitions(alpha-miner PRIVATE ALPHA_HAS_HIP=1)
    target_link_libraries(alpha-miner PRIVATE alpha_hip)
    # Ensure object is rebuilt before linking alpha-miner
    add_dependencies(alpha-miner alpha_hip)
    message(STATUS "HIP backend: ENABLED (${HIPCC})")
  else()
    message(STATUS "HIP backend: disabled (hipcc not found)")
  endif()
endif()

# --- CUDA multi-GPU (NVIDIA) ---
if(ALPHA_MINER_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    target_sources(alpha-miner PRIVATE src/cuda_miner.cu)
    target_compile_definitions(alpha-miner PRIVATE ALPHA_HAS_CUDA=1)
    set_target_properties(alpha-miner PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_ARCHITECTURES "75;80;86;89;90"
    )
    # Ensure host C++ objects can link CUDA runtime
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      target_link_libraries(alpha-miner PRIVATE CUDA::cudart)
    else()
      target_link_libraries(alpha-miner PRIVATE cudart)
    endif()
    message(STATUS "CUDA backend: ENABLED (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "CUDA backend: disabled (nvcc not found)")
  endif()
endif()

install(TARGETS alpha-miner RUNTIME DESTINATION bin)
install(FILES kernels/blake3_an.cl DESTINATION share/alpha-miner/kernels)
