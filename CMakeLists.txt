cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MAMBASERVE_WITH_CUDA "Build Cuda ops/ CudaAllocator" OFF)
option(MAMBASERVE_WITH_NCCL "Build NCCL transport backend" OFF)
option(MAMBASERVE_WITH_NIXL "Build NIXL transport backend" OFF)

# CMAKE_CUDA_ARCHITECTURES must be set before project(... CUDA) / enable_language(CUDA),
# otherwise CMake fills a default (e.g. 52) and a later set is ignored.
#   50 = Maxwell (GTX 750 Ti); 75 = Turing; 80/86 = Ampere; 89 = Ada; 90 = Hopper.
# Override with -DCMAKE_CUDA_ARCHITECTURES=... (e.g. "native" on CMake 3.24+).
if(MAMBASERVE_WITH_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50)
  endif()
  project(MambaServe VERSION 0.0.1 LANGUAGES CXX CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_package(CUDAToolkit REQUIRED)
else()
  project(MambaServe VERSION 0.0.1 LANGUAGES CXX)
endif()

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    safetensors_cpp
    GIT_REPOSITORY https://github.com/syoyo/safetensors-cpp.git
    GIT_TAG main
)
set(SAFETENSORS_CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SAFETENSORS_CPP_BUILD_C_API OFF CACHE BOOL "" FORCE)
set(SAFETENSORS_CPP_CXX_EXCEPTIONS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(safetensors_cpp)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_Populate(eigen)

add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})

FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG main
)
FetchContent_MakeAvailable(tokenizers_cpp)

add_library(mambaserve_common INTERFACE)

target_include_directories(mambaserve_common INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mambaserve_common INTERFACE cxx_std_20)

# For Core library including tensors and status
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "src/core/*.cpp"
)
add_library(core ${CORE_SOURCES})
target_link_libraries(core PUBLIC mambaserve_common)
if(MAMBASERVE_WITH_CUDA)
  target_sources(core PRIVATE src/core/cuda_allocator.cu)
  target_link_libraries(core PUBLIC CUDA::cudart)
  target_compile_definitions(core PUBLIC MAMBASERVE_WITH_CUDA=1)
endif()


# For IO library for reading files, creating cfg and loading weights
file(GLOB_RECURSE IO_SOURCES CONFIGURE_DEPENDS
    "src/io/*.cpp"
)
add_library(io ${IO_SOURCES})
target_link_libraries(io 
    PUBLIC 
        mambaserve_common
        core
    PRIVATE
        nlohmann_json::nlohmann_json
        safetensors-cpp::safetensors_cpp
        
)
if(MAMBASERVE_WITH_CUDA)
  target_compile_definitions(io PRIVATE MAMBASERVE_WITH_CUDA=1)
  target_link_libraries(io PRIVATE CUDA::cudart)
endif()


# Mamba2 Model specific library
file(GLOB_RECURSE MODEL_SOURCES CONFIGURE_DEPENDS
    "src/model/*.cpp"
)
add_library(model ${MODEL_SOURCES})
target_link_libraries(model 
    PUBLIC 
        mambaserve_common
        io
    PRIVATE 
        safetensors-cpp::safetensors_cpp
)
target_compile_definitions(model PRIVATE SAFETENSORS_CPP_NO_IMPLEMENTATION)

# Runtime Library holds actual logic to run the model across different runtimes - cpu, gpu, multi-gpu etc.
file(GLOB_RECURSE RUNTIME_SOURCES CONFIGURE_DEPENDS
    "src/runtime/*.cpp"
)
file(GLOB_RECURSE COMM_SOURCES CONFIGURE_DEPENDS
    "src/comm/*.cpp"
)
add_library(runtime ${RUNTIME_SOURCES} ${COMM_SOURCES})
target_link_libraries(runtime
    PUBLIC 
        mambaserve_common
        model
        io
        core
        ops
    PRIVATE 
        tokenizers_cpp
        safetensors-cpp::safetensors_cpp
)

if(MAMBASERVE_WITH_NCCL)
  if(NOT MAMBASERVE_WITH_CUDA)
    message(FATAL_ERROR "MAMBASERVE_WITH_NCCL requires MAMBASERVE_WITH_CUDA=ON")
  endif()
  find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include)
  find_library(NCCL_LIBRARY nccl
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64)
  if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NCCL not found (set NCCL_ROOT or install libnccl)")
  endif()
  target_include_directories(runtime PRIVATE ${NCCL_INCLUDE_DIR})
  target_link_libraries(runtime PUBLIC ${NCCL_LIBRARY} CUDA::cudart)
  target_compile_definitions(runtime PUBLIC MAMBASERVE_WITH_NCCL=1)
else()
  target_compile_definitions(runtime PUBLIC MAMBASERVE_WITH_NCCL=0)
endif()

if(MAMBASERVE_WITH_NIXL)
  if(NOT MAMBASERVE_WITH_CUDA)
    message(FATAL_ERROR "MAMBASERVE_WITH_NIXL requires MAMBASERVE_WITH_CUDA=ON")
  endif()
  find_path(NIXL_INCLUDE_DIR nixl.h
    HINTS ${NIXL_ROOT} $ENV{NIXL_ROOT} /usr/local /usr
    PATH_SUFFIXES include)
  find_library(NIXL_LIBRARY nixl
    HINTS ${NIXL_ROOT} $ENV{NIXL_ROOT} /usr/local /usr
    PATH_SUFFIXES lib lib64)
  if(NOT NIXL_INCLUDE_DIR OR NOT NIXL_LIBRARY)
    message(FATAL_ERROR "NIXL not found (set NIXL_ROOT or install libnixl)")
  endif()
  target_include_directories(runtime PRIVATE ${NIXL_INCLUDE_DIR})
  target_link_libraries(runtime PUBLIC ${NIXL_LIBRARY} CUDA::cudart)

  # NIXL often needs UCX at link/runtime; pull it in when present.
  find_library(UCX_UCP_LIBRARY ucp
    HINTS ${UCX_ROOT} $ENV{UCX_ROOT} ${NIXL_ROOT} $ENV{NIXL_ROOT} /usr/local /usr
    PATH_SUFFIXES lib lib64)
  find_library(UCX_UCS_LIBRARY ucs
    HINTS ${UCX_ROOT} $ENV{UCX_ROOT} ${NIXL_ROOT} $ENV{NIXL_ROOT} /usr/local /usr
    PATH_SUFFIXES lib lib64)
  if(UCX_UCP_LIBRARY)
    target_link_libraries(runtime PUBLIC ${UCX_UCP_LIBRARY})
  endif()
  if(UCX_UCS_LIBRARY)
    target_link_libraries(runtime PUBLIC ${UCX_UCS_LIBRARY})
  endif()

  target_compile_definitions(runtime PUBLIC MAMBASERVE_WITH_NIXL=1)
else()
  target_compile_definitions(runtime PUBLIC MAMBASERVE_WITH_NIXL=0)
endif()

file(GLOB_RECURSE OPS_CPU_SOURCES CONFIGURE_DEPENDS
    "src/ops/*.cpp"
)
add_library(ops_cpu ${OPS_CPU_SOURCES})
target_link_libraries(ops_cpu
    PUBLIC
        mambaserve_common
        core
        Eigen3::Eigen
)

if(MAMBASERVE_WITH_CUDA)
    file(GLOB_RECURSE OPS_CUDA_SOURCES CONFIGURE_DEPENDS
        "src/ops/cuda/*.cu"
        "src/ops/gpu/*.cu"
        "src/ops/*.cu"
    )
    # No CUDA_SEPARABLE_COMPILATION: cross-TU __device__ helpers live as inline
    # in headers (see block_reductions.cuh). RDC was causing
    # __cudaRegisterLinkedBinary link failures for app/tests.
    add_library(ops_cuda ${OPS_CUDA_SOURCES})
    target_link_libraries(ops_cuda
        PUBLIC
            mambaserve_common
            core
            Eigen3::Eigen
            CUDA::cudart
            CUDA::cublas
    )
    target_compile_options(ops_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    )
    if(TARGET core)
      target_compile_options(core PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
      )
    endif()
endif()

## Umbrella facade for all the different ops implementation
add_library(ops INTERFACE)
target_link_libraries(ops INTERFACE ops_cpu)
if(MAMBASERVE_WITH_CUDA)
    target_link_libraries(ops INTERFACE ops_cuda)
endif()

option(BUILD_TESTING "Build unit tests" OFF)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

option(BUILD_BENCHMARKS "Build Google Benchmark binaries" OFF)

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

add_executable(app
    src/main.cpp
)

target_link_libraries(app
    PRIVATE
        core
        io
        model
        runtime
        safetensors-cpp::safetensors_cpp
        ops
)