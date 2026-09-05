cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
    MambaServe,
    VERSION 0.0.1
    LANGUAGES CXX
)
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

add_library(mambaserve_common INTERFACE)

target_include_directories(mambaserve_common INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mambaserve_common INTERFACE cxx_std_20)

# For Core library including tensors and status
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "src/core/*.cpp"
)
add_library(core ${CORE_SOURCES})
target_link_libraries(core PUBLIC mambaserve_common)


# For IO library for reading files, creating cfg and loading weights
file(GLOB_RECURSE IO_SOURCES CONFIGURE_DEPENDS
    "src/io/*.cpp"
)
add_library(io ${IO_SOURCES})
target_link_libraries(io 
    PUBLIC 
        mambaserve_common
    PRIVATE
        nlohmann_json::nlohmann_json
        safetensors-cpp::safetensors_cpp
        
)


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
add_library(runtime ${RUNTIME_SOURCES})
target_link_libraries(runtime
    PUBLIC 
        mambaserve_common
        model
        io
        core
        ops
    PRIVATE 
        safetensors-cpp::safetensors_cpp
)
target_compile_definitions(model PRIVATE SAFETENSORS_CPP_NO_IMPLEMENTATION)

file(GLOB_RECURSE OPS_CPU_SOURCES CONFIGURE_DEPENDS
    "src/ops/cpu/*.cpp"
)
add_library(ops_cpu ${OPS_CPU_SOURCES})
target_link_libraries(ops_cpu
    PUBLIC
        mambaserve_common
        core
        Eigen3::Eigen
)

## Umbrella facade for all the different ops implementation
add_library(ops INTERFACE)
target_link_libraries(ops INTERFACE ops_cpu)

option(BUILD_TESTING "Build unit tests" OFF)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
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