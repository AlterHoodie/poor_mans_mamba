include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(benches
    generate_bench.cpp
)

target_link_libraries(benches
    PRIVATE
        core
        io
        model
        runtime
        safetensors-cpp::safetensors_cpp
        ops
        benchmark::benchmark
)

target_compile_definitions(benches PRIVATE
    MAMBA_BENCH_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/mamba2-130m-hf"
)

if(MAMBASERVE_WITH_CUDA)
    target_compile_definitions(benches PRIVATE MAMBASERVE_WITH_CUDA=1)
    target_link_libraries(benches PRIVATE CUDA::cudart)
endif()

# ---- transport / cluster drivers (plain mains, CSV output, no google-benchmark) ----
foreach(drv transport_bench cluster_bench)
    add_executable(${drv} ${drv}.cpp)
    target_include_directories(${drv} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${drv}
        PRIVATE
            core
            io
            model
            runtime
            telemetry
            safetensors-cpp::safetensors_cpp
            ops
            Threads::Threads
    )
    if(MAMBASERVE_WITH_CUDA)
        target_compile_definitions(${drv} PRIVATE MAMBASERVE_WITH_CUDA=1)
        target_link_libraries(${drv} PRIVATE CUDA::cudart)
    endif()
endforeach()

# NCCL xfer under GEMM saturation (needs CUDA + cuBLAS; NCCL checked at runtime/compile).
add_executable(nccl_compute_bench nccl_compute_bench.cpp)
target_include_directories(nccl_compute_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nccl_compute_bench
    PRIVATE
        core
        io
        model
        runtime
        telemetry
        safetensors-cpp::safetensors_cpp
        ops
        Threads::Threads
)
if(MAMBASERVE_WITH_CUDA)
    target_compile_definitions(nccl_compute_bench PRIVATE MAMBASERVE_WITH_CUDA=1)
    target_link_libraries(nccl_compute_bench PRIVATE CUDA::cudart CUDA::cublas)
endif()
