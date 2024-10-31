find_package(Protobuf REQUIRED)

# cc_binary(@native_rules_test_repo//:bb)
add_executable(CMakeProject_bb "")
add_executable(CMakeProject::bb ALIAS CMakeProject_bb)
target_link_libraries(CMakeProject_bb PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_bb PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_bb PUBLIC cxx_std_17)
target_sources(CMakeProject_bb PRIVATE
        "${PROJECT_SOURCE_DIR}/a.cc")

# filegroup(@native_rules_test_repo//:subdir_z)
add_library(CMakeProject_subdir_z INTERFACE)
target_sources(CMakeProject_subdir_z INTERFACE
    "${TEST_SRCDIR}/subdir/z.proto")
target_include_directories(CMakeProject_subdir_z INTERFACE
    "${PROJECT_SOURCE_DIR}")

# genrule(@native_rules_test_repo//:h_file)
add_custom_command(
  OUTPUT
    "${TEST_BINDIR}/a.h"
  DEPENDS
    "${TEST_SRCDIR}/x.h"
    "CMakeProject::bb"
  COMMAND $<TARGET_FILE:CMakeProject::bb> ./a .  -I${TEST_BINDIR}/foo -Isubdir/../..  "x.h" "${TEST_BINDIR}/a.h"
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(genrule__CMakeProject_h_file DEPENDS
    "${TEST_BINDIR}/a.h")
add_library(CMakeProject_h_file INTERFACE)
target_sources(CMakeProject_h_file INTERFACE
    "${TEST_BINDIR}/a.h")
target_include_directories(CMakeProject_h_file INTERFACE
    "${PROJECT_BINARY_DIR}")
add_dependencies(CMakeProject_h_file genrule__CMakeProject_h_file)

# cc_library(@native_rules_test_repo//:a)
add_library(CMakeProject_a)
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_include_directories(CMakeProject_a PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a "CMakeProject_h_file")
target_sources(CMakeProject_a PRIVATE
        "${PROJECT_SOURCE_DIR}/a.cc")
add_library(CMakeProject::a ALIAS CMakeProject_a)

# alias(@native_rules_test_repo//:a_alias)
add_library(CMakeProject_a_alias ALIAS CMakeProject_a)
add_library(CMakeProject::a_alias ALIAS CMakeProject_a)

# proto_library(@native_rules_test_repo//:c_proto)
add_library(CMakeProject_c_proto INTERFACE)
target_sources(CMakeProject_c_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/c.proto")
target_include_directories(CMakeProject_c_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
target_link_libraries(CMakeProject_c_proto INTERFACE
    "Protobuf_timestamp_proto")
add_library(CMakeProject::c_proto ALIAS CMakeProject_c_proto)

# @native_rules_test_repo//:aspect_cpp__2c7be24c
# genproto cpp @native_rules_test_repo//:c.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_cpp")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_cpp/c.pb.cc"
    "${TEST_BINDIR}/_gen_cpp/c.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:Protobuf_timestamp_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--cpp_out=${PROJECT_BINARY_DIR}/_gen_cpp"
    "${TEST_SRCDIR}/c.proto"
DEPENDS
    "${TEST_SRCDIR}/c.proto"
    "protobuf::protoc"
COMMENT "Running protoc cpp on ${TEST_SRCDIR}/c.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_cpp__2c7be24c DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.cc")

# @native_rules_test_repo//:c_proto__cpp_library
# aspect cpp @native_rules_test_repo//:c_proto
add_library(CMakeProject_c_proto__cpp_library)
set_property(TARGET CMakeProject_c_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__cpp_library PUBLIC
        "Protobuf::timestamp_proto__cpp_library"
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_c_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_c_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_c_proto__cpp_library "CMakeProject_aspect_cpp__2c7be24c")
target_sources(CMakeProject_c_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.h")
target_sources(CMakeProject_c_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.cc")
add_library(CMakeProject::c_proto__cpp_library ALIAS CMakeProject_c_proto__cpp_library)

# cc_proto_library(@native_rules_test_repo//:c_proto_cc)
add_library(CMakeProject_c_proto_cc)
set_property(TARGET CMakeProject_c_proto_cc PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto_cc PUBLIC
        "CMakeProject::c_proto"
        "CMakeProject::c_proto__cpp_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_c_proto_cc PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto_cc PUBLIC cxx_std_17)
target_sources(CMakeProject_c_proto_cc PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")
add_library(CMakeProject::c_proto_cc ALIAS CMakeProject_c_proto_cc)

# cc_test(@native_rules_test_repo//:a_test)
add_executable(CMakeProject_a_test "")
add_executable(CMakeProject::a_test ALIAS CMakeProject_a_test)
target_link_libraries(CMakeProject_a_test PUBLIC
        "CMakeProject::c_proto_cc"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_test PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_test PUBLIC cxx_std_17)
target_sources(CMakeProject_a_test PRIVATE
        "${PROJECT_SOURCE_DIR}/a.cc")
add_test(NAME CMakeProject_a_test
      COMMAND CMakeProject_a_test
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# cc_library(@native_rules_test_repo//:b)
add_library(CMakeProject_b)
set_property(TARGET CMakeProject_b PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b PUBLIC
        "Threads::Threads"
        "m")
target_link_libraries(CMakeProject_b PRIVATE
        "CMakeProject::a")
target_include_directories(CMakeProject_b PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_b PUBLIC cxx_std_17)
target_sources(CMakeProject_b PRIVATE
        "${PROJECT_SOURCE_DIR}/b.cc")
add_library(CMakeProject::b ALIAS CMakeProject_b)

# proto_library(@native_rules_test_repo//:c_proto_2)
add_library(CMakeProject_c_proto_2 INTERFACE)
target_sources(CMakeProject_c_proto_2 INTERFACE
    "${PROJECT_SOURCE_DIR}/c.proto")
target_include_directories(CMakeProject_c_proto_2 INTERFACE
    "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::c_proto_2 ALIAS CMakeProject_c_proto_2)

# @native_rules_test_repo//:c_proto_2__cpp_library
# aspect cpp @native_rules_test_repo//:c_proto_2
add_library(CMakeProject_c_proto_2__cpp_library)
set_property(TARGET CMakeProject_c_proto_2__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto_2__cpp_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_c_proto_2__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_c_proto_2__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_c_proto_2__cpp_library "CMakeProject_aspect_cpp__2c7be24c")
target_sources(CMakeProject_c_proto_2__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.h")
target_sources(CMakeProject_c_proto_2__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.cc")
add_library(CMakeProject::c_proto_2__cpp_library ALIAS CMakeProject_c_proto_2__cpp_library)

# alias(@native_rules_test_repo//:c_proto_alias)
add_library(CMakeProject_c_proto_alias ALIAS CMakeProject_c_proto)
add_library(CMakeProject::c_proto_alias ALIAS CMakeProject_c_proto)

# alias(@native_rules_test_repo//:c_proto_cc_alias)
add_library(CMakeProject_c_proto_cc_alias ALIAS CMakeProject_c_proto_cc)
add_library(CMakeProject::c_proto_cc_alias ALIAS CMakeProject_c_proto_cc)

# cc_library(@native_rules_test_repo//:subdir_x)
add_library(CMakeProject_subdir_x)
set_property(TARGET CMakeProject_subdir_x PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_x PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_subdir_x PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_subdir_x PUBLIC cxx_std_17)
target_sources(CMakeProject_subdir_x PRIVATE
        "${PROJECT_SOURCE_DIR}/subdir/x.cc")
add_library(CMakeProject::subdir_x ALIAS CMakeProject_subdir_x)

# genrule(@native_rules_test_repo//subdir:make_ycc)
file(MAKE_DIRECTORY "${TEST_BINDIR}/subdir")
add_custom_command(
  OUTPUT
    "${TEST_BINDIR}/subdir/y.cc"
  DEPENDS
    "CMakeProject::bb"
  COMMAND $<TARGET_FILE:CMakeProject::bb> "${TEST_BINDIR}/subdir/y.cc"
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(genrule__CMakeProject_subdir_make_ycc DEPENDS
    "${TEST_BINDIR}/subdir/y.cc")
add_library(CMakeProject_subdir_make_ycc INTERFACE)
target_sources(CMakeProject_subdir_make_ycc INTERFACE
    "${TEST_BINDIR}/subdir/y.cc")
target_include_directories(CMakeProject_subdir_make_ycc INTERFACE
    "${PROJECT_BINARY_DIR}")
add_dependencies(CMakeProject_subdir_make_ycc genrule__CMakeProject_subdir_make_ycc)

# genrule(@native_rules_test_repo//subdir:make_y)
file(MAKE_DIRECTORY "${TEST_BINDIR}/subdir")
add_custom_command(
  OUTPUT
    "${TEST_BINDIR}/subdir/y.h"
  DEPENDS
    "CMakeProject::bb"
  COMMAND $<TARGET_FILE:CMakeProject::bb> "${TEST_BINDIR}/subdir/y.h"
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(genrule__CMakeProject_subdir_make_y DEPENDS
    "${TEST_BINDIR}/subdir/y.h")
add_library(CMakeProject_subdir_make_y INTERFACE)
target_sources(CMakeProject_subdir_make_y INTERFACE
    "${TEST_BINDIR}/subdir/y.h")
target_include_directories(CMakeProject_subdir_make_y INTERFACE
    "${PROJECT_BINARY_DIR}")
add_dependencies(CMakeProject_subdir_make_y genrule__CMakeProject_subdir_make_y)

# cc_library(@native_rules_test_repo//subdir:y)
add_library(CMakeProject_subdir_y)
set_property(TARGET CMakeProject_subdir_y PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_y PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_subdir_y PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_include_directories(CMakeProject_subdir_y PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_subdir_y PUBLIC cxx_std_17)
add_dependencies(CMakeProject_subdir_y "CMakeProject_subdir_make_y" "CMakeProject_subdir_make_ycc")
target_sources(CMakeProject_subdir_y PRIVATE
        "${PROJECT_BINARY_DIR}/subdir/y.cc")
add_library(CMakeProject::subdir_y ALIAS CMakeProject_subdir_y)

# alias(@native_rules_test_repo//:y_alias)
add_library(CMakeProject_y_alias ALIAS CMakeProject_subdir_y)
add_library(CMakeProject::y_alias ALIAS CMakeProject_subdir_y)

# cc_library(@native_rules_test_repo//subdir:y_include_prefix)
add_library(CMakeProject_subdir_y_include_prefix)
set_property(TARGET CMakeProject_subdir_y_include_prefix PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_y_include_prefix PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_subdir_y_include_prefix PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_include_directories(CMakeProject_subdir_y_include_prefix PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_subdir_y_include_prefix PUBLIC cxx_std_17)
add_dependencies(CMakeProject_subdir_y_include_prefix "CMakeProject_subdir_make_y" "CMakeProject_subdir_make_ycc")
target_sources(CMakeProject_subdir_y_include_prefix PRIVATE
        "${PROJECT_BINARY_DIR}/subdir/y.cc")
add_library(CMakeProject::subdir_y_include_prefix ALIAS CMakeProject_subdir_y_include_prefix)

# cc_library(@native_rules_test_repo//subdir:y_includes)
add_library(CMakeProject_subdir_y_includes)
set_property(TARGET CMakeProject_subdir_y_includes PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_y_includes PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_subdir_y_includes PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/subdir>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/subdir>")
target_include_directories(CMakeProject_subdir_y_includes PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_subdir_y_includes PUBLIC cxx_std_17)
add_dependencies(CMakeProject_subdir_y_includes "CMakeProject_subdir_make_y" "CMakeProject_subdir_make_ycc")
target_sources(CMakeProject_subdir_y_includes PRIVATE
        "${PROJECT_BINARY_DIR}/subdir/y.cc")
add_library(CMakeProject::subdir_y_includes ALIAS CMakeProject_subdir_y_includes)

# cc_library(@native_rules_test_repo//subdir:y_strip_include_prefix)
add_library(CMakeProject_subdir_y_strip_include_prefix)
set_property(TARGET CMakeProject_subdir_y_strip_include_prefix PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_y_strip_include_prefix PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_subdir_y_strip_include_prefix PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/subdir>")
target_include_directories(CMakeProject_subdir_y_strip_include_prefix PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_subdir_y_strip_include_prefix PUBLIC cxx_std_17)
add_dependencies(CMakeProject_subdir_y_strip_include_prefix "CMakeProject_subdir_make_y" "CMakeProject_subdir_make_ycc")
target_sources(CMakeProject_subdir_y_strip_include_prefix PRIVATE
        "${PROJECT_BINARY_DIR}/subdir/y.cc")
add_library(CMakeProject::subdir_y_strip_include_prefix ALIAS CMakeProject_subdir_y_strip_include_prefix)

# proto_library(@native_rules_test_repo//subdir:z_proto)
add_library(CMakeProject_subdir_z_proto INTERFACE)
target_sources(CMakeProject_subdir_z_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/subdir/z.proto")
target_include_directories(CMakeProject_subdir_z_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::subdir_z_proto ALIAS CMakeProject_subdir_z_proto)

# @native_rules_test_repo//subdir:aspect_cpp__2eff1b8c
# genproto cpp @native_rules_test_repo//subdir:z.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_cpp/subdir")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_cpp/subdir/z.pb.cc"
    "${TEST_BINDIR}/_gen_cpp/subdir/z.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_subdir_z_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--cpp_out=${PROJECT_BINARY_DIR}/_gen_cpp"
    "${TEST_SRCDIR}/subdir/z.proto"
DEPENDS
    "${TEST_SRCDIR}/subdir/z.proto"
    "protobuf::protoc"
COMMENT "Running protoc cpp on ${TEST_SRCDIR}/subdir/z.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_subdir_aspect_cpp__2eff1b8c DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/subdir/z.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/subdir/z.pb.cc")

# @native_rules_test_repo//subdir:z_proto__cpp_library
# aspect cpp @native_rules_test_repo//subdir:z_proto
add_library(CMakeProject_subdir_z_proto__cpp_library)
set_property(TARGET CMakeProject_subdir_z_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_z_proto__cpp_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_subdir_z_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_subdir_z_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_subdir_z_proto__cpp_library "CMakeProject_subdir_aspect_cpp__2eff1b8c")
target_sources(CMakeProject_subdir_z_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/subdir/z.pb.h")
target_sources(CMakeProject_subdir_z_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/subdir/z.pb.cc")
add_library(CMakeProject::subdir_z_proto__cpp_library ALIAS CMakeProject_subdir_z_proto__cpp_library)

# cc_library(@native_rules_test_repo//:defines)
add_library(CMakeProject_defines)
set_property(TARGET CMakeProject_defines PROPERTY LINKER_LANGUAGE "CXX")
target_compile_definitions(CMakeProject_defines PUBLIC "VERSION=\"\"")
target_link_libraries(CMakeProject_defines PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_defines PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_include_directories(CMakeProject_defines PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_defines PUBLIC cxx_std_17)
add_dependencies(CMakeProject_defines "CMakeProject_h_file")
target_sources(CMakeProject_defines PRIVATE
        "${PROJECT_SOURCE_DIR}/a.cc")
add_library(CMakeProject::defines ALIAS CMakeProject_defines)
