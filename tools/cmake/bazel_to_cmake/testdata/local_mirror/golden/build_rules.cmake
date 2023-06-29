# Loading local_proto_mirror
add_subdirectory("_cmake_binary_dir_/_local_mirror/lpm-src" "_cmake_binary_dir_/_local_mirror/lpm-build" EXCLUDE_FROM_ALL)
find_package(lpm REQUIRED)

# cc_library(@local_mirror_test_repo//:a)
add_library(CMakeProject_a)
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
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
add_library(CMakeProject::a ALIAS CMakeProject_a)
