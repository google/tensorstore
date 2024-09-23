find_package(Protobuf REQUIRED)

# proto_library(@rules_proto_test_repo//:a_proto)
add_library(CMakeProject_a_proto INTERFACE)
target_sources(CMakeProject_a_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/a.proto")
target_include_directories(CMakeProject_a_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::a_proto ALIAS CMakeProject_a_proto)

# @rules_proto_test_repo//:aspect_cpp__43b0bcf1
# genproto cpp @rules_proto_test_repo//:a.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_cpp")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_cpp/a.pb.cc"
    "${TEST_BINDIR}/_gen_cpp/a.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_a_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--cpp_out=${PROJECT_BINARY_DIR}/_gen_cpp"
    "${TEST_SRCDIR}/a.proto"
DEPENDS
    "${TEST_SRCDIR}/a.proto"
    "protobuf::protoc"
COMMENT "Running protoc cpp on ${TEST_SRCDIR}/a.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_cpp__43b0bcf1 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.cc")

# @rules_proto_test_repo//:a_proto__cpp_library
# aspect cpp @rules_proto_test_repo//:a_proto
add_library(CMakeProject_a_proto__cpp_library)
set_property(TARGET CMakeProject_a_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__cpp_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_a_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_a_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a_proto__cpp_library "CMakeProject_aspect_cpp__43b0bcf1")
target_sources(CMakeProject_a_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.h")
target_sources(CMakeProject_a_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.cc")
add_library(CMakeProject::a_proto__cpp_library ALIAS CMakeProject_a_proto__cpp_library)

# @rules_proto_test_repo//:aspect_upb_minitable__43b0bcf1
# genproto upb_minitable @rules_proto_test_repo//:a.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb_minitable")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb_minitable/a.upb_minitable.c"
    "${TEST_BINDIR}/_gen_upb_minitable/a.upb_minitable.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_a_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_minitable_out=${PROJECT_BINARY_DIR}/_gen_upb_minitable"
    "${TEST_SRCDIR}/a.proto"
DEPENDS
    "${TEST_SRCDIR}/a.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb_minitable_stage1"
COMMENT "Running protoc upb_minitable on ${TEST_SRCDIR}/a.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb_minitable__43b0bcf1 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.h"
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.c")

# @rules_proto_test_repo//:a_proto__minitable_library
# aspect upb_minitable @rules_proto_test_repo//:a_proto
add_library(CMakeProject_a_proto__minitable_library)
set_property(TARGET CMakeProject_a_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__minitable_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_a_proto__minitable_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb_minitable>")
target_compile_features(CMakeProject_a_proto__minitable_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a_proto__minitable_library "CMakeProject_aspect_upb_minitable__43b0bcf1")
target_sources(CMakeProject_a_proto__minitable_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.h")
target_sources(CMakeProject_a_proto__minitable_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.c")
add_library(CMakeProject::a_proto__minitable_library ALIAS CMakeProject_a_proto__minitable_library)

# @rules_proto_test_repo//:aspect_upb__43b0bcf1
# genproto upb @rules_proto_test_repo//:a.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb/a.upb.c"
    "${TEST_BINDIR}/_gen_upb/a.upb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_a_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_out=${PROJECT_BINARY_DIR}/_gen_upb"
    "${TEST_SRCDIR}/a.proto"
DEPENDS
    "${TEST_SRCDIR}/a.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb"
COMMENT "Running protoc upb on ${TEST_SRCDIR}/a.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb__43b0bcf1 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.h"
    "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.c")

# @rules_proto_test_repo//:a_proto__upb_library
# aspect upb @rules_proto_test_repo//:a_proto
add_library(CMakeProject_a_proto__upb_library)
set_property(TARGET CMakeProject_a_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__upb_library PUBLIC
        "CMakeProject::a_proto__minitable_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_a_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb>")
target_compile_features(CMakeProject_a_proto__upb_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a_proto__upb_library "CMakeProject_aspect_upb__43b0bcf1")
target_sources(CMakeProject_a_proto__upb_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.h")
target_sources(CMakeProject_a_proto__upb_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.c")
add_library(CMakeProject::a_proto__upb_library ALIAS CMakeProject_a_proto__upb_library)

# @rules_proto_test_repo//:aspect_upbdefs__43b0bcf1
# genproto upbdefs @rules_proto_test_repo//:a.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upbdefs")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upbdefs/a.upbdefs.c"
    "${TEST_BINDIR}/_gen_upbdefs/a.upbdefs.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_a_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upbdefs_out=${PROJECT_BINARY_DIR}/_gen_upbdefs"
    "${TEST_SRCDIR}/a.proto"
DEPENDS
    "${TEST_SRCDIR}/a.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upbdefs"
COMMENT "Running protoc upbdefs on ${TEST_SRCDIR}/a.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upbdefs__43b0bcf1 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.h"
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.c")

# @rules_proto_test_repo//:a_proto__upbdefs_library
# aspect upbdefs @rules_proto_test_repo//:a_proto
add_library(CMakeProject_a_proto__upbdefs_library)
set_property(TARGET CMakeProject_a_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__upbdefs_library PUBLIC
        "CMakeProject::a_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upbdefs>")
target_compile_features(CMakeProject_a_proto__upbdefs_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a_proto__upbdefs_library "CMakeProject_aspect_upbdefs__43b0bcf1")
target_sources(CMakeProject_a_proto__upbdefs_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.h")
target_sources(CMakeProject_a_proto__upbdefs_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.c")
add_library(CMakeProject::a_proto__upbdefs_library ALIAS CMakeProject_a_proto__upbdefs_library)

# proto_library(@rules_proto_test_repo//:ab_proto)
add_library(CMakeProject_ab_proto INTERFACE)
target_sources(CMakeProject_ab_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/a.proto"
    "${PROJECT_SOURCE_DIR}/b.proto")
target_include_directories(CMakeProject_ab_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::ab_proto ALIAS CMakeProject_ab_proto)

# @rules_proto_test_repo//:aspect_cpp__3daadd2b
# genproto cpp @rules_proto_test_repo//:b.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_cpp")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_cpp/b.pb.cc"
    "${TEST_BINDIR}/_gen_cpp/b.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_ab_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--cpp_out=${PROJECT_BINARY_DIR}/_gen_cpp"
    "${TEST_SRCDIR}/b.proto"
DEPENDS
    "${TEST_SRCDIR}/b.proto"
    "protobuf::protoc"
COMMENT "Running protoc cpp on ${TEST_SRCDIR}/b.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_cpp__3daadd2b DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/b.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/b.pb.cc")

# @rules_proto_test_repo//:ab_proto__cpp_library
# aspect cpp @rules_proto_test_repo//:ab_proto
add_library(CMakeProject_ab_proto__cpp_library)
set_property(TARGET CMakeProject_ab_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_ab_proto__cpp_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_ab_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_ab_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_ab_proto__cpp_library "CMakeProject_aspect_cpp__3daadd2b" "CMakeProject_aspect_cpp__43b0bcf1")
target_sources(CMakeProject_ab_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.h"
        "${PROJECT_BINARY_DIR}/_gen_cpp/b.pb.h")
target_sources(CMakeProject_ab_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.cc"
        "${PROJECT_BINARY_DIR}/_gen_cpp/b.pb.cc")
add_library(CMakeProject::ab_proto__cpp_library ALIAS CMakeProject_ab_proto__cpp_library)

# @rules_proto_test_repo//:aspect_upb__3daadd2b
# genproto upb @rules_proto_test_repo//:b.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb/b.upb.c"
    "${TEST_BINDIR}/_gen_upb/b.upb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_ab_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_out=${PROJECT_BINARY_DIR}/_gen_upb"
    "${TEST_SRCDIR}/b.proto"
DEPENDS
    "${TEST_SRCDIR}/b.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb"
COMMENT "Running protoc upb on ${TEST_SRCDIR}/b.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb__3daadd2b DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb/b.upb.h"
    "${PROJECT_BINARY_DIR}/_gen_upb/b.upb.c")

# @rules_proto_test_repo//:aspect_upb_minitable__3daadd2b
# genproto upb_minitable @rules_proto_test_repo//:b.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb_minitable")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb_minitable/b.upb_minitable.c"
    "${TEST_BINDIR}/_gen_upb_minitable/b.upb_minitable.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_ab_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_minitable_out=${PROJECT_BINARY_DIR}/_gen_upb_minitable"
    "${TEST_SRCDIR}/b.proto"
DEPENDS
    "${TEST_SRCDIR}/b.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb_minitable_stage1"
COMMENT "Running protoc upb_minitable on ${TEST_SRCDIR}/b.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb_minitable__3daadd2b DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/b.upb_minitable.h"
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/b.upb_minitable.c")

# @rules_proto_test_repo//:ab_proto__minitable_library
# aspect upb_minitable @rules_proto_test_repo//:ab_proto
add_library(CMakeProject_ab_proto__minitable_library)
set_property(TARGET CMakeProject_ab_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_ab_proto__minitable_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_ab_proto__minitable_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb_minitable>")
target_compile_features(CMakeProject_ab_proto__minitable_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_ab_proto__minitable_library "CMakeProject_aspect_upb_minitable__3daadd2b" "CMakeProject_aspect_upb_minitable__43b0bcf1")
target_sources(CMakeProject_ab_proto__minitable_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.h"
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/b.upb_minitable.h")
target_sources(CMakeProject_ab_proto__minitable_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.c"
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/b.upb_minitable.c")
add_library(CMakeProject::ab_proto__minitable_library ALIAS CMakeProject_ab_proto__minitable_library)

# @rules_proto_test_repo//:ab_proto__upb_library
# aspect upb @rules_proto_test_repo//:ab_proto
add_library(CMakeProject_ab_proto__upb_library)
set_property(TARGET CMakeProject_ab_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_ab_proto__upb_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_ab_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb>")
target_compile_features(CMakeProject_ab_proto__upb_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_ab_proto__upb_library "CMakeProject_aspect_upb__3daadd2b" "CMakeProject_aspect_upb__43b0bcf1")
target_sources(CMakeProject_ab_proto__upb_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.h"
        "${PROJECT_BINARY_DIR}/_gen_upb/b.upb.h")
target_sources(CMakeProject_ab_proto__upb_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.c"
        "${PROJECT_BINARY_DIR}/_gen_upb/b.upb.c")
add_library(CMakeProject::ab_proto__upb_library ALIAS CMakeProject_ab_proto__upb_library)

# @rules_proto_test_repo//:aspect_upbdefs__3daadd2b
# genproto upbdefs @rules_proto_test_repo//:b.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upbdefs")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upbdefs/b.upbdefs.c"
    "${TEST_BINDIR}/_gen_upbdefs/b.upbdefs.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_ab_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upbdefs_out=${PROJECT_BINARY_DIR}/_gen_upbdefs"
    "${TEST_SRCDIR}/b.proto"
DEPENDS
    "${TEST_SRCDIR}/b.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upbdefs"
COMMENT "Running protoc upbdefs on ${TEST_SRCDIR}/b.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upbdefs__3daadd2b DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/b.upbdefs.h"
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/b.upbdefs.c")

# @rules_proto_test_repo//:ab_proto__upbdefs_library
# aspect upbdefs @rules_proto_test_repo//:ab_proto
add_library(CMakeProject_ab_proto__upbdefs_library)
set_property(TARGET CMakeProject_ab_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_ab_proto__upbdefs_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_ab_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upbdefs>")
target_compile_features(CMakeProject_ab_proto__upbdefs_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_ab_proto__upbdefs_library "CMakeProject_aspect_upbdefs__3daadd2b" "CMakeProject_aspect_upbdefs__43b0bcf1")
target_sources(CMakeProject_ab_proto__upbdefs_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.h"
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/b.upbdefs.h")
target_sources(CMakeProject_ab_proto__upbdefs_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.c"
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/b.upbdefs.c")
add_library(CMakeProject::ab_proto__upbdefs_library ALIAS CMakeProject_ab_proto__upbdefs_library)

# cc_proto_library(@rules_proto_test_repo//:ab_protos_cc)
add_library(CMakeProject_ab_protos_cc)
set_property(TARGET CMakeProject_ab_protos_cc PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_ab_protos_cc PUBLIC
        "CMakeProject::ab_proto"
        "CMakeProject::ab_proto__cpp_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_ab_protos_cc PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_ab_protos_cc PUBLIC cxx_std_17)
target_sources(CMakeProject_ab_protos_cc PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")
add_library(CMakeProject::ab_protos_cc ALIAS CMakeProject_ab_protos_cc)

# proto_library(@rules_proto_test_repo//:d_proto)
add_library(CMakeProject_d_proto INTERFACE)
target_sources(CMakeProject_d_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/d.proto")
target_include_directories(CMakeProject_d_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::d_proto ALIAS CMakeProject_d_proto)

# proto_library(@rules_proto_test_repo//:abcd_proto)
add_library(CMakeProject_abcd_proto INTERFACE)
target_sources(CMakeProject_abcd_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/c.proto")
target_include_directories(CMakeProject_abcd_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
target_link_libraries(CMakeProject_abcd_proto INTERFACE
    "CMakeProject::ab_proto"
    "CMakeProject::d_proto")
add_library(CMakeProject::abcd_proto ALIAS CMakeProject_abcd_proto)

# @rules_proto_test_repo//:aspect_cpp__84819b61
# genproto cpp @rules_proto_test_repo//:c.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_cpp")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_cpp/c.pb.cc"
    "${TEST_BINDIR}/_gen_cpp/c.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject::ab_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject::d_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_abcd_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--cpp_out=${PROJECT_BINARY_DIR}/_gen_cpp"
    "${TEST_SRCDIR}/c.proto"
DEPENDS
    "${TEST_SRCDIR}/c.proto"
    "protobuf::protoc"
COMMENT "Running protoc cpp on ${TEST_SRCDIR}/c.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_cpp__84819b61 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.cc")

# @rules_proto_test_repo//:aspect_cpp__ae7c401d
# genproto cpp @rules_proto_test_repo//:d.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_cpp")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_cpp/d.pb.cc"
    "${TEST_BINDIR}/_gen_cpp/d.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_d_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--cpp_out=${PROJECT_BINARY_DIR}/_gen_cpp"
    "${TEST_SRCDIR}/d.proto"
DEPENDS
    "${TEST_SRCDIR}/d.proto"
    "protobuf::protoc"
COMMENT "Running protoc cpp on ${TEST_SRCDIR}/d.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_cpp__ae7c401d DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/d.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/d.pb.cc")

# @rules_proto_test_repo//:d_proto__cpp_library
# aspect cpp @rules_proto_test_repo//:d_proto
add_library(CMakeProject_d_proto__cpp_library)
set_property(TARGET CMakeProject_d_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_d_proto__cpp_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_d_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_d_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_d_proto__cpp_library "CMakeProject_aspect_cpp__ae7c401d")
target_sources(CMakeProject_d_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/d.pb.h")
target_sources(CMakeProject_d_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/d.pb.cc")
add_library(CMakeProject::d_proto__cpp_library ALIAS CMakeProject_d_proto__cpp_library)

# @rules_proto_test_repo//:abcd_proto__cpp_library
# aspect cpp @rules_proto_test_repo//:abcd_proto
add_library(CMakeProject_abcd_proto__cpp_library)
set_property(TARGET CMakeProject_abcd_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abcd_proto__cpp_library PUBLIC
        "CMakeProject::ab_proto__cpp_library"
        "CMakeProject::d_proto__cpp_library"
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_abcd_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_abcd_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_abcd_proto__cpp_library "CMakeProject_aspect_cpp__84819b61")
target_sources(CMakeProject_abcd_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.h")
target_sources(CMakeProject_abcd_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.cc")
add_library(CMakeProject::abcd_proto__cpp_library ALIAS CMakeProject_abcd_proto__cpp_library)

# @rules_proto_test_repo//:aspect_upb__84819b61
# genproto upb @rules_proto_test_repo//:c.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb/c.upb.c"
    "${TEST_BINDIR}/_gen_upb/c.upb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject::ab_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject::d_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_abcd_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_out=${PROJECT_BINARY_DIR}/_gen_upb"
    "${TEST_SRCDIR}/c.proto"
DEPENDS
    "${TEST_SRCDIR}/c.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb"
COMMENT "Running protoc upb on ${TEST_SRCDIR}/c.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb__84819b61 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb/c.upb.h"
    "${PROJECT_BINARY_DIR}/_gen_upb/c.upb.c")

# @rules_proto_test_repo//:aspect_upb__ae7c401d
# genproto upb @rules_proto_test_repo//:d.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb/d.upb.c"
    "${TEST_BINDIR}/_gen_upb/d.upb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_d_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_out=${PROJECT_BINARY_DIR}/_gen_upb"
    "${TEST_SRCDIR}/d.proto"
DEPENDS
    "${TEST_SRCDIR}/d.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb"
COMMENT "Running protoc upb on ${TEST_SRCDIR}/d.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb__ae7c401d DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb/d.upb.h"
    "${PROJECT_BINARY_DIR}/_gen_upb/d.upb.c")

# @rules_proto_test_repo//:aspect_upb_minitable__ae7c401d
# genproto upb_minitable @rules_proto_test_repo//:d.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb_minitable")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb_minitable/d.upb_minitable.c"
    "${TEST_BINDIR}/_gen_upb_minitable/d.upb_minitable.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_d_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_minitable_out=${PROJECT_BINARY_DIR}/_gen_upb_minitable"
    "${TEST_SRCDIR}/d.proto"
DEPENDS
    "${TEST_SRCDIR}/d.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb_minitable_stage1"
COMMENT "Running protoc upb_minitable on ${TEST_SRCDIR}/d.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb_minitable__ae7c401d DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/d.upb_minitable.h"
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/d.upb_minitable.c")

# @rules_proto_test_repo//:d_proto__minitable_library
# aspect upb_minitable @rules_proto_test_repo//:d_proto
add_library(CMakeProject_d_proto__minitable_library)
set_property(TARGET CMakeProject_d_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_d_proto__minitable_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_d_proto__minitable_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb_minitable>")
target_compile_features(CMakeProject_d_proto__minitable_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_d_proto__minitable_library "CMakeProject_aspect_upb_minitable__ae7c401d")
target_sources(CMakeProject_d_proto__minitable_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/d.upb_minitable.h")
target_sources(CMakeProject_d_proto__minitable_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/d.upb_minitable.c")
add_library(CMakeProject::d_proto__minitable_library ALIAS CMakeProject_d_proto__minitable_library)

# @rules_proto_test_repo//:d_proto__upb_library
# aspect upb @rules_proto_test_repo//:d_proto
add_library(CMakeProject_d_proto__upb_library)
set_property(TARGET CMakeProject_d_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_d_proto__upb_library PUBLIC
        "CMakeProject::d_proto__minitable_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_d_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb>")
target_compile_features(CMakeProject_d_proto__upb_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_d_proto__upb_library "CMakeProject_aspect_upb__ae7c401d")
target_sources(CMakeProject_d_proto__upb_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb/d.upb.h")
target_sources(CMakeProject_d_proto__upb_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb/d.upb.c")
add_library(CMakeProject::d_proto__upb_library ALIAS CMakeProject_d_proto__upb_library)

# @rules_proto_test_repo//:aspect_upb_minitable__84819b61
# genproto upb_minitable @rules_proto_test_repo//:c.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb_minitable")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb_minitable/c.upb_minitable.c"
    "${TEST_BINDIR}/_gen_upb_minitable/c.upb_minitable.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject::ab_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject::d_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_abcd_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_minitable_out=${PROJECT_BINARY_DIR}/_gen_upb_minitable"
    "${TEST_SRCDIR}/c.proto"
DEPENDS
    "${TEST_SRCDIR}/c.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb_minitable_stage1"
COMMENT "Running protoc upb_minitable on ${TEST_SRCDIR}/c.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb_minitable__84819b61 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/c.upb_minitable.h"
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/c.upb_minitable.c")

# @rules_proto_test_repo//:abcd_proto__minitable_library
# aspect upb_minitable @rules_proto_test_repo//:abcd_proto
add_library(CMakeProject_abcd_proto__minitable_library)
set_property(TARGET CMakeProject_abcd_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abcd_proto__minitable_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "CMakeProject::d_proto__minitable_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_abcd_proto__minitable_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb_minitable>")
target_compile_features(CMakeProject_abcd_proto__minitable_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_abcd_proto__minitable_library "CMakeProject_aspect_upb_minitable__84819b61")
target_sources(CMakeProject_abcd_proto__minitable_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/c.upb_minitable.h")
target_sources(CMakeProject_abcd_proto__minitable_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/c.upb_minitable.c")
add_library(CMakeProject::abcd_proto__minitable_library ALIAS CMakeProject_abcd_proto__minitable_library)

# @rules_proto_test_repo//:abcd_proto__upb_library
# aspect upb @rules_proto_test_repo//:abcd_proto
add_library(CMakeProject_abcd_proto__upb_library)
set_property(TARGET CMakeProject_abcd_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abcd_proto__upb_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "CMakeProject::ab_proto__upb_library"
        "CMakeProject::abcd_proto__minitable_library"
        "CMakeProject::d_proto__minitable_library"
        "CMakeProject::d_proto__upb_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_abcd_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb>")
target_compile_features(CMakeProject_abcd_proto__upb_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_abcd_proto__upb_library "CMakeProject_aspect_upb__84819b61")
target_sources(CMakeProject_abcd_proto__upb_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb/c.upb.h")
target_sources(CMakeProject_abcd_proto__upb_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb/c.upb.c")
add_library(CMakeProject::abcd_proto__upb_library ALIAS CMakeProject_abcd_proto__upb_library)

# @rules_proto_test_repo//:aspect_upbdefs__84819b61
# genproto upbdefs @rules_proto_test_repo//:c.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upbdefs")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upbdefs/c.upbdefs.c"
    "${TEST_BINDIR}/_gen_upbdefs/c.upbdefs.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject::ab_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject::d_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_abcd_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upbdefs_out=${PROJECT_BINARY_DIR}/_gen_upbdefs"
    "${TEST_SRCDIR}/c.proto"
DEPENDS
    "${TEST_SRCDIR}/c.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upbdefs"
COMMENT "Running protoc upbdefs on ${TEST_SRCDIR}/c.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upbdefs__84819b61 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/c.upbdefs.h"
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/c.upbdefs.c")

# @rules_proto_test_repo//:aspect_upbdefs__ae7c401d
# genproto upbdefs @rules_proto_test_repo//:d.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upbdefs")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upbdefs/d.upbdefs.c"
    "${TEST_BINDIR}/_gen_upbdefs/d.upbdefs.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_d_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upbdefs_out=${PROJECT_BINARY_DIR}/_gen_upbdefs"
    "${TEST_SRCDIR}/d.proto"
DEPENDS
    "${TEST_SRCDIR}/d.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upbdefs"
COMMENT "Running protoc upbdefs on ${TEST_SRCDIR}/d.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upbdefs__ae7c401d DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/d.upbdefs.h"
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/d.upbdefs.c")

# @rules_proto_test_repo//:d_proto__upbdefs_library
# aspect upbdefs @rules_proto_test_repo//:d_proto
add_library(CMakeProject_d_proto__upbdefs_library)
set_property(TARGET CMakeProject_d_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_d_proto__upbdefs_library PUBLIC
        "CMakeProject::d_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_d_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upbdefs>")
target_compile_features(CMakeProject_d_proto__upbdefs_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_d_proto__upbdefs_library "CMakeProject_aspect_upbdefs__ae7c401d")
target_sources(CMakeProject_d_proto__upbdefs_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/d.upbdefs.h")
target_sources(CMakeProject_d_proto__upbdefs_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/d.upbdefs.c")
add_library(CMakeProject::d_proto__upbdefs_library ALIAS CMakeProject_d_proto__upbdefs_library)

# @rules_proto_test_repo//:abcd_proto__upbdefs_library
# aspect upbdefs @rules_proto_test_repo//:abcd_proto
add_library(CMakeProject_abcd_proto__upbdefs_library)
set_property(TARGET CMakeProject_abcd_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abcd_proto__upbdefs_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "CMakeProject::ab_proto__upbdefs_library"
        "CMakeProject::abcd_proto__minitable_library"
        "CMakeProject::d_proto__minitable_library"
        "CMakeProject::d_proto__upbdefs_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_abcd_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upbdefs>")
target_compile_features(CMakeProject_abcd_proto__upbdefs_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_abcd_proto__upbdefs_library "CMakeProject_aspect_upbdefs__84819b61")
target_sources(CMakeProject_abcd_proto__upbdefs_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/c.upbdefs.h")
target_sources(CMakeProject_abcd_proto__upbdefs_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/c.upbdefs.c")
add_library(CMakeProject::abcd_proto__upbdefs_library ALIAS CMakeProject_abcd_proto__upbdefs_library)

# cc_proto_library(@rules_proto_test_repo//:abcd_protos_cc)
add_library(CMakeProject_abcd_protos_cc)
set_property(TARGET CMakeProject_abcd_protos_cc PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abcd_protos_cc PUBLIC
        "CMakeProject::abcd_proto"
        "CMakeProject::abcd_proto__cpp_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_abcd_protos_cc PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abcd_protos_cc PUBLIC cxx_std_17)
target_sources(CMakeProject_abcd_protos_cc PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")
add_library(CMakeProject::abcd_protos_cc ALIAS CMakeProject_abcd_protos_cc)

# proto_library(@rules_proto_test_repo//:b_proto)
add_library(CMakeProject_b_proto INTERFACE)
target_sources(CMakeProject_b_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/b.proto")
target_include_directories(CMakeProject_b_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::b_proto ALIAS CMakeProject_b_proto)

# @rules_proto_test_repo//:b_proto__cpp_library
# aspect cpp @rules_proto_test_repo//:b_proto
add_library(CMakeProject_b_proto__cpp_library)
set_property(TARGET CMakeProject_b_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_proto__cpp_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_b_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_b_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_b_proto__cpp_library "CMakeProject_aspect_cpp__3daadd2b")
target_sources(CMakeProject_b_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/b.pb.h")
target_sources(CMakeProject_b_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/b.pb.cc")
add_library(CMakeProject::b_proto__cpp_library ALIAS CMakeProject_b_proto__cpp_library)

# @rules_proto_test_repo//:b_proto__minitable_library
# aspect upb_minitable @rules_proto_test_repo//:b_proto
add_library(CMakeProject_b_proto__minitable_library)
set_property(TARGET CMakeProject_b_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_proto__minitable_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_b_proto__minitable_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb_minitable>")
target_compile_features(CMakeProject_b_proto__minitable_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_b_proto__minitable_library "CMakeProject_aspect_upb_minitable__3daadd2b")
target_sources(CMakeProject_b_proto__minitable_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/b.upb_minitable.h")
target_sources(CMakeProject_b_proto__minitable_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/b.upb_minitable.c")
add_library(CMakeProject::b_proto__minitable_library ALIAS CMakeProject_b_proto__minitable_library)

# @rules_proto_test_repo//:b_proto__upb_library
# aspect upb @rules_proto_test_repo//:b_proto
add_library(CMakeProject_b_proto__upb_library)
set_property(TARGET CMakeProject_b_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_proto__upb_library PUBLIC
        "CMakeProject::b_proto__minitable_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_b_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb>")
target_compile_features(CMakeProject_b_proto__upb_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_b_proto__upb_library "CMakeProject_aspect_upb__3daadd2b")
target_sources(CMakeProject_b_proto__upb_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb/b.upb.h")
target_sources(CMakeProject_b_proto__upb_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb/b.upb.c")
add_library(CMakeProject::b_proto__upb_library ALIAS CMakeProject_b_proto__upb_library)

# @rules_proto_test_repo//:b_proto__upbdefs_library
# aspect upbdefs @rules_proto_test_repo//:b_proto
add_library(CMakeProject_b_proto__upbdefs_library)
set_property(TARGET CMakeProject_b_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_proto__upbdefs_library PUBLIC
        "CMakeProject::b_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_b_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upbdefs>")
target_compile_features(CMakeProject_b_proto__upbdefs_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_b_proto__upbdefs_library "CMakeProject_aspect_upbdefs__3daadd2b")
target_sources(CMakeProject_b_proto__upbdefs_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/b.upbdefs.h")
target_sources(CMakeProject_b_proto__upbdefs_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/b.upbdefs.c")
add_library(CMakeProject::b_proto__upbdefs_library ALIAS CMakeProject_b_proto__upbdefs_library)

# proto_library(@rules_proto_test_repo//:x_proto)
add_library(CMakeProject_x_proto INTERFACE)
target_sources(CMakeProject_x_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/x.proto")
target_include_directories(CMakeProject_x_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
target_link_libraries(CMakeProject_x_proto INTERFACE
    "Protobuf_any_proto")
add_library(CMakeProject::x_proto ALIAS CMakeProject_x_proto)

# @rules_proto_test_repo//:aspect_upb__a1bf1338
# genproto upb @rules_proto_test_repo//:x.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb/x.upb.c"
    "${TEST_BINDIR}/_gen_upb/x.upb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_x_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:Protobuf_any_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_out=${PROJECT_BINARY_DIR}/_gen_upb"
    "${TEST_SRCDIR}/x.proto"
DEPENDS
    "${TEST_SRCDIR}/x.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb"
COMMENT "Running protoc upb on ${TEST_SRCDIR}/x.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb__a1bf1338 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb/x.upb.h"
    "${PROJECT_BINARY_DIR}/_gen_upb/x.upb.c")

# @rules_proto_test_repo//:aspect_upb_minitable__a1bf1338
# genproto upb_minitable @rules_proto_test_repo//:x.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb_minitable")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb_minitable/x.upb_minitable.c"
    "${TEST_BINDIR}/_gen_upb_minitable/x.upb_minitable.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_x_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:Protobuf_any_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_minitable_out=${PROJECT_BINARY_DIR}/_gen_upb_minitable"
    "${TEST_SRCDIR}/x.proto"
DEPENDS
    "${TEST_SRCDIR}/x.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb_minitable_stage1"
COMMENT "Running protoc upb_minitable on ${TEST_SRCDIR}/x.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb_minitable__a1bf1338 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/x.upb_minitable.h"
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/x.upb_minitable.c")

# @rules_proto_test_repo//:x_proto__minitable_library
# aspect upb_minitable @rules_proto_test_repo//:x_proto
add_library(CMakeProject_x_proto__minitable_library)
set_property(TARGET CMakeProject_x_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto__minitable_library PUBLIC
        "Protobuf::any_proto__minitable_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_x_proto__minitable_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb_minitable>")
target_compile_features(CMakeProject_x_proto__minitable_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_x_proto__minitable_library "CMakeProject_aspect_upb_minitable__a1bf1338")
target_sources(CMakeProject_x_proto__minitable_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/x.upb_minitable.h")
target_sources(CMakeProject_x_proto__minitable_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/x.upb_minitable.c")
add_library(CMakeProject::x_proto__minitable_library ALIAS CMakeProject_x_proto__minitable_library)

# @rules_proto_test_repo//:x_proto__upb_library
# aspect upb @rules_proto_test_repo//:x_proto
add_library(CMakeProject_x_proto__upb_library)
set_property(TARGET CMakeProject_x_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto__upb_library PUBLIC
        "CMakeProject::x_proto__minitable_library"
        "Protobuf::any_proto__minitable_library"
        "Protobuf::any_proto__upb_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_x_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb>")
target_compile_features(CMakeProject_x_proto__upb_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_x_proto__upb_library "CMakeProject_aspect_upb__a1bf1338")
target_sources(CMakeProject_x_proto__upb_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb/x.upb.h")
target_sources(CMakeProject_x_proto__upb_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb/x.upb.c")
add_library(CMakeProject::x_proto__upb_library ALIAS CMakeProject_x_proto__upb_library)

# @rules_proto_test_repo//:aspect_upbdefs__a1bf1338
# genproto upbdefs @rules_proto_test_repo//:x.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upbdefs")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upbdefs/x.upbdefs.c"
    "${TEST_BINDIR}/_gen_upbdefs/x.upbdefs.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_x_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:Protobuf_any_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upbdefs_out=${PROJECT_BINARY_DIR}/_gen_upbdefs"
    "${TEST_SRCDIR}/x.proto"
DEPENDS
    "${TEST_SRCDIR}/x.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upbdefs"
COMMENT "Running protoc upbdefs on ${TEST_SRCDIR}/x.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upbdefs__a1bf1338 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/x.upbdefs.h"
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/x.upbdefs.c")

# @rules_proto_test_repo//:x_proto__upbdefs_library
# aspect upbdefs @rules_proto_test_repo//:x_proto
add_library(CMakeProject_x_proto__upbdefs_library)
set_property(TARGET CMakeProject_x_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto__upbdefs_library PUBLIC
        "CMakeProject::x_proto__minitable_library"
        "Protobuf::any_proto__minitable_library"
        "Protobuf::any_proto__upbdefs_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_x_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upbdefs>")
target_compile_features(CMakeProject_x_proto__upbdefs_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_x_proto__upbdefs_library "CMakeProject_aspect_upbdefs__a1bf1338")
target_sources(CMakeProject_x_proto__upbdefs_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/x.upbdefs.h")
target_sources(CMakeProject_x_proto__upbdefs_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/x.upbdefs.c")
add_library(CMakeProject::x_proto__upbdefs_library ALIAS CMakeProject_x_proto__upbdefs_library)

# @rules_proto_test_repo//:aspect_cpp__a1bf1338
# genproto cpp @rules_proto_test_repo//:x.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_cpp")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_cpp/x.pb.cc"
    "${TEST_BINDIR}/_gen_cpp/x.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_x_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "-I$<JOIN:$<TARGET_PROPERTY:Protobuf_any_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--cpp_out=${PROJECT_BINARY_DIR}/_gen_cpp"
    "${TEST_SRCDIR}/x.proto"
DEPENDS
    "${TEST_SRCDIR}/x.proto"
    "protobuf::protoc"
COMMENT "Running protoc cpp on ${TEST_SRCDIR}/x.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_cpp__a1bf1338 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/x.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/x.pb.cc")

# @rules_proto_test_repo//:x_proto__cpp_library
# aspect cpp @rules_proto_test_repo//:x_proto
add_library(CMakeProject_x_proto__cpp_library)
set_property(TARGET CMakeProject_x_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto__cpp_library PUBLIC
        "Protobuf::any_proto__cpp_library"
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_x_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_x_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_x_proto__cpp_library "CMakeProject_aspect_cpp__a1bf1338")
target_sources(CMakeProject_x_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/x.pb.h")
target_sources(CMakeProject_x_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/x.pb.cc")
add_library(CMakeProject::x_proto__cpp_library ALIAS CMakeProject_x_proto__cpp_library)

# cc_proto_library(@rules_proto_test_repo//:x_proto_cc)
add_library(CMakeProject_x_proto_cc)
set_property(TARGET CMakeProject_x_proto_cc PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto_cc PUBLIC
        "CMakeProject::x_proto"
        "CMakeProject::x_proto__cpp_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_x_proto_cc PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_x_proto_cc PUBLIC cxx_std_17)
target_sources(CMakeProject_x_proto_cc PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")
add_library(CMakeProject::x_proto_cc ALIAS CMakeProject_x_proto_cc)

# proto_library(@rules_proto_test_repo//:y_proto)
add_library(CMakeProject_y_proto INTERFACE)
target_sources(CMakeProject_y_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/src/subdir/y.proto")
target_include_directories(CMakeProject_y_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/src")
add_library(CMakeProject::y_proto ALIAS CMakeProject_y_proto)

# @rules_proto_test_repo//:aspect_cpp__9651e148
# genproto cpp @rules_proto_test_repo//:src/subdir/y.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_cpp/src/subdir")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_cpp/src/subdir/y.pb.cc"
    "${TEST_BINDIR}/_gen_cpp/src/subdir/y.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_y_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--cpp_out=${PROJECT_BINARY_DIR}/_gen_cpp/src"
    "${TEST_SRCDIR}/src/subdir/y.proto"
DEPENDS
    "${TEST_SRCDIR}/src/subdir/y.proto"
    "protobuf::protoc"
COMMENT "Running protoc cpp on ${TEST_SRCDIR}/src/subdir/y.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_cpp__9651e148 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/src/subdir/y.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/src/subdir/y.pb.cc")

# @rules_proto_test_repo//:y_proto__cpp_library
# aspect cpp @rules_proto_test_repo//:y_proto
add_library(CMakeProject_y_proto__cpp_library)
set_property(TARGET CMakeProject_y_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_y_proto__cpp_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_y_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp/src>")
target_compile_features(CMakeProject_y_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_y_proto__cpp_library "CMakeProject_aspect_cpp__9651e148")
target_sources(CMakeProject_y_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/src/subdir/y.pb.h")
target_sources(CMakeProject_y_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/src/subdir/y.pb.cc")
add_library(CMakeProject::y_proto__cpp_library ALIAS CMakeProject_y_proto__cpp_library)

# @rules_proto_test_repo//:aspect_upb__9651e148
# genproto upb @rules_proto_test_repo//:src/subdir/y.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb/src/subdir")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb/src/subdir/y.upb.c"
    "${TEST_BINDIR}/_gen_upb/src/subdir/y.upb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_y_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_out=${PROJECT_BINARY_DIR}/_gen_upb/src"
    "${TEST_SRCDIR}/src/subdir/y.proto"
DEPENDS
    "${TEST_SRCDIR}/src/subdir/y.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb"
COMMENT "Running protoc upb on ${TEST_SRCDIR}/src/subdir/y.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb__9651e148 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb/src/subdir/y.upb.h"
    "${PROJECT_BINARY_DIR}/_gen_upb/src/subdir/y.upb.c")

# @rules_proto_test_repo//:aspect_upb_minitable__9651e148
# genproto upb_minitable @rules_proto_test_repo//:src/subdir/y.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb_minitable/src/subdir")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb_minitable/src/subdir/y.upb_minitable.c"
    "${TEST_BINDIR}/_gen_upb_minitable/src/subdir/y.upb_minitable.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_y_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--upb_minitable_out=${PROJECT_BINARY_DIR}/_gen_upb_minitable/src"
    "${TEST_SRCDIR}/src/subdir/y.proto"
DEPENDS
    "${TEST_SRCDIR}/src/subdir/y.proto"
    "protobuf::protoc"
    "protobuf::protoc_gen_upb_minitable_stage1"
COMMENT "Running protoc upb_minitable on ${TEST_SRCDIR}/src/subdir/y.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_upb_minitable__9651e148 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/src/subdir/y.upb_minitable.h"
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/src/subdir/y.upb_minitable.c")

# @rules_proto_test_repo//:y_proto__minitable_library
# aspect upb_minitable @rules_proto_test_repo//:y_proto
add_library(CMakeProject_y_proto__minitable_library)
set_property(TARGET CMakeProject_y_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_y_proto__minitable_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_y_proto__minitable_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb_minitable/src>")
target_compile_features(CMakeProject_y_proto__minitable_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_y_proto__minitable_library "CMakeProject_aspect_upb_minitable__9651e148")
target_sources(CMakeProject_y_proto__minitable_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/src/subdir/y.upb_minitable.h")
target_sources(CMakeProject_y_proto__minitable_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/src/subdir/y.upb_minitable.c")
add_library(CMakeProject::y_proto__minitable_library ALIAS CMakeProject_y_proto__minitable_library)

# @rules_proto_test_repo//:y_proto__upb_library
# aspect upb @rules_proto_test_repo//:y_proto
add_library(CMakeProject_y_proto__upb_library)
set_property(TARGET CMakeProject_y_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_y_proto__upb_library PUBLIC
        "CMakeProject::y_proto__minitable_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_y_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb/src>")
target_compile_features(CMakeProject_y_proto__upb_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_y_proto__upb_library "CMakeProject_aspect_upb__9651e148")
target_sources(CMakeProject_y_proto__upb_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb/src/subdir/y.upb.h")
target_sources(CMakeProject_y_proto__upb_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb/src/subdir/y.upb.c")
add_library(CMakeProject::y_proto__upb_library ALIAS CMakeProject_y_proto__upb_library)
