find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# proto_library(@grpc_generate_cc_test_repo//:c_proto)
add_library(CMakeProject_c_proto INTERFACE)
target_sources(CMakeProject_c_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/c.proto")
target_include_directories(CMakeProject_c_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::c_proto ALIAS CMakeProject_c_proto)

# @grpc_generate_cc_test_repo//:aspect_upb__71950a86
# genproto upb @grpc_generate_cc_test_repo//:c.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb/c.upb.c"
    "${TEST_BINDIR}/_gen_upb/c.upb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
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
add_custom_target(CMakeProject_aspect_upb__71950a86 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb/c.upb.h"
    "${PROJECT_BINARY_DIR}/_gen_upb/c.upb.c")

# @grpc_generate_cc_test_repo//:aspect_upb_minitable__71950a86
# genproto upb_minitable @grpc_generate_cc_test_repo//:c.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upb_minitable")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upb_minitable/c.upb_minitable.c"
    "${TEST_BINDIR}/_gen_upb_minitable/c.upb_minitable.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
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
add_custom_target(CMakeProject_aspect_upb_minitable__71950a86 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/c.upb_minitable.h"
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/c.upb_minitable.c")

# @grpc_generate_cc_test_repo//:c_proto__minitable_library
# aspect upb_minitable @grpc_generate_cc_test_repo//:c_proto
add_library(CMakeProject_c_proto__minitable_library)
set_property(TARGET CMakeProject_c_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__minitable_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_c_proto__minitable_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb_minitable>")
target_compile_features(CMakeProject_c_proto__minitable_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_c_proto__minitable_library "CMakeProject_aspect_upb_minitable__71950a86")
target_sources(CMakeProject_c_proto__minitable_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/c.upb_minitable.h")
target_sources(CMakeProject_c_proto__minitable_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/c.upb_minitable.c")
add_library(CMakeProject::c_proto__minitable_library ALIAS CMakeProject_c_proto__minitable_library)

# @grpc_generate_cc_test_repo//:c_proto__upb_library
# aspect upb @grpc_generate_cc_test_repo//:c_proto
add_library(CMakeProject_c_proto__upb_library)
set_property(TARGET CMakeProject_c_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upb_library PUBLIC
        "CMakeProject::c_proto__minitable_library"
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_c_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb>")
target_compile_features(CMakeProject_c_proto__upb_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_c_proto__upb_library "CMakeProject_aspect_upb__71950a86")
target_sources(CMakeProject_c_proto__upb_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb/c.upb.h")
target_sources(CMakeProject_c_proto__upb_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb/c.upb.c")
add_library(CMakeProject::c_proto__upb_library ALIAS CMakeProject_c_proto__upb_library)

# @grpc_generate_cc_test_repo//:cc__grpc_codegenfile(MAKE_DIRECTORY "${TEST_BINDIR}")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/c.grpc.pb.cc"
    "${TEST_BINDIR}/c.grpc.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--grpc_out=services_namespace=grpc_gen:${PROJECT_BINARY_DIR}"
    "${TEST_SRCDIR}/c.proto"
DEPENDS
    "${TEST_SRCDIR}/c.proto"
    "gRPC::grpc_cpp_plugin"
    "protobuf::protoc"
COMMENT "Running protoc grpc on ${TEST_SRCDIR}/c.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_cc__grpc_codegen DEPENDS
  "${TEST_BINDIR}/c.grpc.pb.h"
    "${TEST_BINDIR}/c.grpc.pb.cc")

# cc_library(@grpc_generate_cc_test_repo//:cc_grpc)
add_library(CMakeProject_cc_grpc)
set_property(TARGET CMakeProject_cc_grpc PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_cc_grpc PUBLIC
        "Threads::Threads"
        "gRPC::gRPC_codegen"
        "m")
target_include_directories(CMakeProject_cc_grpc PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_include_directories(CMakeProject_cc_grpc PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_cc_grpc PUBLIC cxx_std_17)
target_sources(CMakeProject_cc_grpc PRIVATE
        "${PROJECT_BINARY_DIR}/c.grpc.pb.cc")
add_library(CMakeProject::cc_grpc ALIAS CMakeProject_cc_grpc)

# cc_library(@grpc_generate_cc_test_repo//:a)
add_library(CMakeProject_a)
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "CMakeProject::cc_grpc"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
target_sources(CMakeProject_a PRIVATE
        "${PROJECT_SOURCE_DIR}/a.cc")
add_library(CMakeProject::a ALIAS CMakeProject_a)

# @grpc_generate_cc_test_repo//:aspect_cpp__71950a86
# genproto cpp @grpc_generate_cc_test_repo//:c.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_cpp")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_cpp/c.pb.cc"
    "${TEST_BINDIR}/_gen_cpp/c.pb.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
    "--cpp_out=${PROJECT_BINARY_DIR}/_gen_cpp"
    "${TEST_SRCDIR}/c.proto"
DEPENDS
    "${TEST_SRCDIR}/c.proto"
    "protobuf::protoc"
COMMENT "Running protoc cpp on ${TEST_SRCDIR}/c.proto"
COMMAND_EXPAND_LISTS
VERBATIM
)
add_custom_target(CMakeProject_aspect_cpp__71950a86 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.cc")

# @grpc_generate_cc_test_repo//:c_proto__cpp_library
# aspect cpp @grpc_generate_cc_test_repo//:c_proto
add_library(CMakeProject_c_proto__cpp_library)
set_property(TARGET CMakeProject_c_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__cpp_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_c_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_c_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_c_proto__cpp_library "CMakeProject_aspect_cpp__71950a86")
target_sources(CMakeProject_c_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.h")
target_sources(CMakeProject_c_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/c.pb.cc")
add_library(CMakeProject::c_proto__cpp_library ALIAS CMakeProject_c_proto__cpp_library)

# @grpc_generate_cc_test_repo//:aspect_upbdefs__71950a86
# genproto upbdefs @grpc_generate_cc_test_repo//:c.proto
file(MAKE_DIRECTORY "${TEST_BINDIR}/_gen_upbdefs")
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/_gen_upbdefs/c.upbdefs.c"
    "${TEST_BINDIR}/_gen_upbdefs/c.upbdefs.h"
COMMAND $<TARGET_FILE:protobuf::protoc>
    --experimental_allow_proto3_optional
    --plugin=protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,$<SEMICOLON>-I>"
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
add_custom_target(CMakeProject_aspect_upbdefs__71950a86 DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/c.upbdefs.h"
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/c.upbdefs.c")

# @grpc_generate_cc_test_repo//:c_proto__upbdefs_library
# aspect upbdefs @grpc_generate_cc_test_repo//:c_proto
add_library(CMakeProject_c_proto__upbdefs_library)
set_property(TARGET CMakeProject_c_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upbdefs_library PUBLIC
        "CMakeProject::c_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_c_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upbdefs>")
target_compile_features(CMakeProject_c_proto__upbdefs_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_c_proto__upbdefs_library "CMakeProject_aspect_upbdefs__71950a86")
target_sources(CMakeProject_c_proto__upbdefs_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/c.upbdefs.h")
target_sources(CMakeProject_c_proto__upbdefs_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/c.upbdefs.c")
add_library(CMakeProject::c_proto__upbdefs_library ALIAS CMakeProject_c_proto__upbdefs_library)
