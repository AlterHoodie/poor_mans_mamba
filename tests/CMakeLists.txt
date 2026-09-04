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
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests
    PRIVATE
        core
        runtime
        ops
        GTest::gtest_main
)

## Cmake functions to discover gtest files in a particular directory
include(GoogleTest)
gtest_discover_tests(tests)
