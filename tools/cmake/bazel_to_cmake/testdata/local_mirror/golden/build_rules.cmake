# Loading local_proto_mirror
file(DOWNLOAD "https://raw.githubusercontent.com/bufbuild/protoc-gen-validate/2682ad06cca00550030e177834f58a2bc06eb61e/validate/validate.proto" "_cmake_binary_dir_/local_mirror/lpm/validate.proto"
     EXPECTED_HASH "SHA256=bf7ca2ac45a75b8b9ff12f38efd7f48ee460ede1a7919d60c93fad3a64fc2eee")

add_subdirectory("_cmake_binary_dir_/local_mirror/lpm" "_cmake_binary_dir_/_build_local_mirror/lpm" EXCLUDE_FROM_ALL)
find_package(lpm REQUIRED)

# @bazel_test_repo//:a
add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "Threads::Threads"
        "lpm::b"
        "lpm::validate_cc"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_library(CMakeProject::a ALIAS CMakeProject_a)
