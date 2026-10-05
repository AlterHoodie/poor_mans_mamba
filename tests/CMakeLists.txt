set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.18.0
)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
    "*.cpp"
)
# GPU op tests require CUDA; keep CPU-only builds free of those sources.
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*/ops/gpu/.*")

if(MAMBASERVE_WITH_CUDA)
    file(GLOB GPU_TEST_SOURCES CONFIGURE_DEPENDS
        "ops/gpu/*.cpp"
        "ops/gpu/*.cu"
    )
    list(APPEND TEST_SOURCES ${GPU_TEST_SOURCES})
endif()

add_executable(tests ${TEST_SOURCES})
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tests
    PRIVATE
        core
        runtime
        ops
        GTest::gtest
)
target_compile_definitions(tests PRIVATE
    MAMBA_TEST_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/mamba2-130m-hf"
    MAMBA_TEST_FALCON_DIR="${PROJECT_SOURCE_DIR}/models/falcon-h1-0.5b-base"
)

if(MAMBASERVE_WITH_CUDA)
    target_link_libraries(tests PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_compile_options(tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    )
endif()

## Cmake functions to discover gtest files in a particular directory
include(GoogleTest)
gtest_discover_tests(tests)
