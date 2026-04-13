include(FetchContent)
option(TENSORSTORE_USE_SYSTEM_BAZEL_SKYLIB "Use an installed version of bazel_skylib")
# Loading bazel_skylib
FetchContent_Declare(bazel_skylib
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy CMakeLists.txt orig_CMakeLists.cmake && ${CMAKE_COMMAND} -E copy "${TEST_BINDIR}/third_party/bazel_skylib-proxy-CMakeLists.txt" CMakeLists.txt
    OVERRIDE_FIND_PACKAGE)
add_subdirectory("${TEST_BINDIR}/third_party" _third_party_configs EXCLUDE_FROM_ALL)
find_package(bazel_skylib REQUIRED)

# cc_library(@bzlmod_simple_test_repo//:main)
add_library(CMakeProject_main)
set_property(TARGET CMakeProject_main PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_main PUBLIC
        "bazel_skylib::vars"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_main PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_main PUBLIC cxx_std_17)
target_sources(CMakeProject_main PRIVATE
        "${PROJECT_SOURCE_DIR}/main.cc")
add_library(CMakeProject::main ALIAS CMakeProject_main)
