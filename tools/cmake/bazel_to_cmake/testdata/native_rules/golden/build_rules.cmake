find_package(Protobuf REQUIRED)

# @bazel_test_repo//:bb
add_executable(CMakeProject_bb "")
add_executable(CMakeProject::bb ALIAS CMakeProject_bb)
target_sources(CMakeProject_bb PRIVATE
        "${TEST_DIRECTORY}/a.cc")
target_link_libraries(CMakeProject_bb PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_bb PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_bb PUBLIC cxx_std_17)

# @bazel_test_repo//:h_file
add_custom_command(
  OUTPUT "_cmake_binary_dir_/a.h"
  DEPENDS "${TEST_DIRECTORY}/x.h" "CMakeProject::bb"
  COMMAND bash -c "$<TARGET_FILE:CMakeProject_bb> $(dirname $(dirname \"x.h\" )) $(dirname \"x.h\" ) \"x.h\" \"_cmake_binary_dir_/a.h\""
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(CMakeProject_h_file DEPENDS "_cmake_binary_dir_/a.h")

# @bazel_test_repo//:a
add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a "CMakeProject_h_file")
add_library(CMakeProject::a ALIAS CMakeProject_a)

# @bazel_test_repo//:a_alias
add_library(CMakeProject_a_alias ALIAS CMakeProject_a)
add_library(CMakeProject::a_alias ALIAS CMakeProject_a)

# @bazel_test_repo//:c_proto
add_library(CMakeProject_c_proto INTERFACE)
target_sources(CMakeProject_c_proto INTERFACE
        "${TEST_DIRECTORY}/c.proto")
btc_transitive_import_dirs(
    OUT_VAR CMakeProject_c_proto_IMPORT_DIRS
    IMPORT_DIRS "${TEST_DIRECTORY}"
    IMPORT_TARGETS Protobuf_timestamp_proto 
    IMPORT_VARS Protobuf_IMPORT_DIRS
)
set_property(TARGET CMakeProject_c_proto PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMakeProject_c_proto_IMPORT_DIRS})

# @bazel_test_repo//:c_proto__cpp_library
add_library(CMakeProject_c_proto__cpp_library)
target_sources(CMakeProject_c_proto__cpp_library PRIVATE
        "${TEST_DIRECTORY}/c.proto")
set_property(TARGET CMakeProject_c_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_c_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto__cpp_library PUBLIC cxx_std_17)

btc_protobuf(
    TARGET CMakeProject_c_proto__cpp_library
    IMPORT_TARGETS  CMakeProject_c_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    DEPENDENCIES "protobuf::protoc"
)

# @bazel_test_repo//:c_proto_cc
add_library(CMakeProject_c_proto_cc INTERFACE)
target_link_libraries(CMakeProject_c_proto_cc INTERFACE
        "CMakeProject_c_proto__cpp_library")
target_include_directories(CMakeProject_c_proto_cc INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto_cc INTERFACE cxx_std_17)
add_library(CMakeProject::c_proto_cc ALIAS CMakeProject_c_proto_cc)

# @bazel_test_repo//:a_test
add_executable(CMakeProject_a_test "")
add_executable(CMakeProject::a_test ALIAS CMakeProject_a_test)
target_sources(CMakeProject_a_test PRIVATE
        "${TEST_DIRECTORY}/a.cc")
target_link_libraries(CMakeProject_a_test PUBLIC
        "CMakeProject::c_proto_cc"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_test PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_test PUBLIC cxx_std_17)
add_test(NAME CMakeProject_a_test COMMAND CMakeProject_a_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# @bazel_test_repo//:c_proto_2
add_library(CMakeProject_c_proto_2 INTERFACE)
target_sources(CMakeProject_c_proto_2 INTERFACE
        "${TEST_DIRECTORY}/c.proto")
list(APPEND CMakeProject_c_proto_2_IMPORT_DIRS "${TEST_DIRECTORY}")
set_property(TARGET CMakeProject_c_proto_2 PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMakeProject_c_proto_2_IMPORT_DIRS})

# @bazel_test_repo//:c_proto_cc_alias
add_library(CMakeProject_c_proto_cc_alias ALIAS CMakeProject_c_proto_cc)
add_library(CMakeProject::c_proto_cc_alias ALIAS CMakeProject_c_proto_cc)
