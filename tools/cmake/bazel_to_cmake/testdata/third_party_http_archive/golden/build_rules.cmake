include(FetchContent)
option(TENSORSTORE_USE_SYSTEM_HALF "Use an installed version of half")
# Loading net_sourceforge_half
FetchContent_Declare(half
    URL "https://storage.googleapis.com/tensorstore-bazel-mirror/sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip"
    URL_HASH "SHA256=ad1788afe0300fa2b02b0d1df128d857f021f92ccf7c8bddd07812685fa07a25"
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy "${TEST_DIRECTORY}/half.BUILD.bazel" BUILD.bazel && ${CMAKE_COMMAND} -E copy "_cmake_binary_dir_/third_party/half-proxy-CMakeLists.txt" CMakeLists.txt
    OVERRIDE_FIND_PACKAGE)
add_subdirectory("_cmake_binary_dir_/third_party" _third_party_configs EXCLUDE_FROM_ALL)
find_package(half REQUIRED)

# @bazel_test_repo//:a
add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "Threads::Threads"
        "half::half"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_library(CMakeProject::a ALIAS CMakeProject_a)
