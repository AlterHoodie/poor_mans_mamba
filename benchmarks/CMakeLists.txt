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
