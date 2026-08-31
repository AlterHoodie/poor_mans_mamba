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

add_library(mambaserve_common INTERFACE)

target_include_directories(mambaserve_common INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mambaserve_common INTERFACE cxx_std_20)

# file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
#     "src/core/*.cpp"
#     "src/model/*.cpp"
# )
# add_library(core ${CORE_SOURCES})
# target_link_libraries(core PUBLIC mambaserve_common)


file(GLOB_RECURSE CONFIG_SOURCES CONFIGURE_DEPENDS
# "src/loader/*.cpp"
"src/parser/*.cpp"
)
add_library(config ${CONFIG_SOURCES})
target_link_libraries(config
    PUBLIC 
        mambaserve_common
    PRIVATE
        nlohmann_json::nlohmann_json)



# target_include_directories(core
#     PUBLIC
#         ${PROJECT_SOURCE_DIR}/include
# )

add_executable(app
    src/main.cpp
)

target_link_libraries(app
    PRIVATE
        # core
        config
)