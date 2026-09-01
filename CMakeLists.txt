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

add_library(mambaserve_common INTERFACE)

target_include_directories(mambaserve_common INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mambaserve_common INTERFACE cxx_std_20)

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "src/core/*.cpp"
)
add_library(core ${CORE_SOURCES})
target_link_libraries(core PUBLIC mambaserve_common)

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "src/io/*.cpp"
)
add_library(io ${CORE_SOURCES})
target_link_libraries(io 
    PUBLIC 
        mambaserve_common
    PRIVATE
        nlohmann_json::nlohmann_json
        safetensors-cpp::safetensors_cpp
        
)

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "src/model/*.cpp"
)
add_library(model ${CORE_SOURCES})
target_link_libraries(model 
    PUBLIC 
        mambaserve_common
        io
    PRIVATE 
        safetensors-cpp::safetensors_cpp
)
target_compile_definitions(model PRIVATE SAFETENSORS_CPP_NO_IMPLEMENTATION)

add_executable(app
    src/main.cpp
)

target_link_libraries(app
    PRIVATE
        core
        io
        model
        safetensors-cpp::safetensors_cpp
)