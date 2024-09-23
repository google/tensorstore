find_package(Protobuf REQUIRED)

# proto_library(@upb_proto_library_test_repo//:a_proto)
add_library(CMakeProject_a_proto INTERFACE)
target_sources(CMakeProject_a_proto INTERFACE
    "${PROJECT_SOURCE_DIR}/a.proto")
target_include_directories(CMakeProject_a_proto INTERFACE
    "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::a_proto ALIAS CMakeProject_a_proto)

# @upb_proto_library_test_repo//:aspect_cpp__9411025a
# genproto cpp @upb_proto_library_test_repo//:a.proto
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
add_custom_target(CMakeProject_aspect_cpp__9411025a DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.h"
    "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.cc")

# @upb_proto_library_test_repo//:a_proto__cpp_library
# aspect cpp @upb_proto_library_test_repo//:a_proto
add_library(CMakeProject_a_proto__cpp_library)
set_property(TARGET CMakeProject_a_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__cpp_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_a_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_cpp>")
target_compile_features(CMakeProject_a_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a_proto__cpp_library "CMakeProject_aspect_cpp__9411025a")
target_sources(CMakeProject_a_proto__cpp_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.h")
target_sources(CMakeProject_a_proto__cpp_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_cpp/a.pb.cc")
add_library(CMakeProject::a_proto__cpp_library ALIAS CMakeProject_a_proto__cpp_library)

# cc_proto_library(@upb_proto_library_test_repo//:a_cc_proto)
add_library(CMakeProject_a_cc_proto)
set_property(TARGET CMakeProject_a_cc_proto PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_cc_proto PUBLIC
        "CMakeProject::a_proto"
        "CMakeProject::a_proto__cpp_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_cc_proto PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_cc_proto PUBLIC cxx_std_17)
target_sources(CMakeProject_a_cc_proto PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")
add_library(CMakeProject::a_cc_proto ALIAS CMakeProject_a_cc_proto)

# @upb_proto_library_test_repo//:aspect_upb__9411025a
# genproto upb @upb_proto_library_test_repo//:a.proto
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
add_custom_target(CMakeProject_aspect_upb__9411025a DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.h"
    "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.c")

# @upb_proto_library_test_repo//:aspect_upb_minitable__9411025a
# genproto upb_minitable @upb_proto_library_test_repo//:a.proto
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
add_custom_target(CMakeProject_aspect_upb_minitable__9411025a DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.h"
    "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.c")

# @upb_proto_library_test_repo//:a_proto__minitable_library
# aspect upb_minitable @upb_proto_library_test_repo//:a_proto
add_library(CMakeProject_a_proto__minitable_library)
set_property(TARGET CMakeProject_a_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__minitable_library PUBLIC
        "Threads::Threads"
        "m"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_a_proto__minitable_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/_gen_upb_minitable>")
target_compile_features(CMakeProject_a_proto__minitable_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a_proto__minitable_library "CMakeProject_aspect_upb_minitable__9411025a")
target_sources(CMakeProject_a_proto__minitable_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.h")
target_sources(CMakeProject_a_proto__minitable_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb_minitable/a.upb_minitable.c")
add_library(CMakeProject::a_proto__minitable_library ALIAS CMakeProject_a_proto__minitable_library)

# @upb_proto_library_test_repo//:a_proto__upb_library
# aspect upb @upb_proto_library_test_repo//:a_proto
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
add_dependencies(CMakeProject_a_proto__upb_library "CMakeProject_aspect_upb__9411025a")
target_sources(CMakeProject_a_proto__upb_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.h")
target_sources(CMakeProject_a_proto__upb_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upb/a.upb.c")
add_library(CMakeProject::a_proto__upb_library ALIAS CMakeProject_a_proto__upb_library)

# upb_c_proto_library(@upb_proto_library_test_repo//:a_upb_proto)
add_library(CMakeProject_a_upb_proto)
set_property(TARGET CMakeProject_a_upb_proto PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_upb_proto PUBLIC
        "CMakeProject::a_proto"
        "CMakeProject::a_proto__minitable_library"
        "CMakeProject::a_proto__upb_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_upb_proto PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_upb_proto PUBLIC cxx_std_17)
target_sources(CMakeProject_a_upb_proto PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")
add_library(CMakeProject::a_upb_proto ALIAS CMakeProject_a_upb_proto)

# @upb_proto_library_test_repo//:aspect_upbdefs__9411025a
# genproto upbdefs @upb_proto_library_test_repo//:a.proto
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
add_custom_target(CMakeProject_aspect_upbdefs__9411025a DEPENDS
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.h"
    "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.c")

# @upb_proto_library_test_repo//:a_proto__upbdefs_library
# aspect upbdefs @upb_proto_library_test_repo//:a_proto
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
add_dependencies(CMakeProject_a_proto__upbdefs_library "CMakeProject_aspect_upbdefs__9411025a")
target_sources(CMakeProject_a_proto__upbdefs_library PUBLIC
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.h")
target_sources(CMakeProject_a_proto__upbdefs_library PRIVATE
        "${PROJECT_BINARY_DIR}/_gen_upbdefs/a.upbdefs.c")
add_library(CMakeProject::a_proto__upbdefs_library ALIAS CMakeProject_a_proto__upbdefs_library)

# upb_proto_reflection_library(@upb_proto_library_test_repo//:a_upb_proto_reflection)
add_library(CMakeProject_a_upb_proto_reflection)
set_property(TARGET CMakeProject_a_upb_proto_reflection PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_upb_proto_reflection PUBLIC
        "CMakeProject::a_proto"
        "CMakeProject::a_proto__minitable_library"
        "CMakeProject::a_proto__upbdefs_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_upb_proto_reflection PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_upb_proto_reflection PUBLIC cxx_std_17)
target_sources(CMakeProject_a_upb_proto_reflection PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")
add_library(CMakeProject::a_upb_proto_reflection ALIAS CMakeProject_a_upb_proto_reflection)

# upb_c_proto_library(@upb_proto_library_test_repo//:abc_upb_proto)
add_library(CMakeProject_abc_upb_proto)
set_property(TARGET CMakeProject_abc_upb_proto PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abc_upb_proto PUBLIC
        "CMakeProject::abc_protos"
        "CMakeProject::abc_protos__minitable_library"
        "CMakeProject::abc_protos__upb_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_abc_upb_proto PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_upb_proto PUBLIC cxx_std_17)
target_sources(CMakeProject_abc_upb_proto PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")
add_library(CMakeProject::abc_upb_proto ALIAS CMakeProject_abc_upb_proto)

# upb_proto_reflection_library(@upb_proto_library_test_repo//:abc_upb_proto_reflection)
add_library(CMakeProject_abc_upb_proto_reflection)
set_property(TARGET CMakeProject_abc_upb_proto_reflection PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abc_upb_proto_reflection PUBLIC
        "CMakeProject::abc_protos"
        "CMakeProject::abc_protos__minitable_library"
        "CMakeProject::abc_protos__upbdefs_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_abc_upb_proto_reflection PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_upb_proto_reflection PUBLIC cxx_std_17)
target_sources(CMakeProject_abc_upb_proto_reflection PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")
add_library(CMakeProject::abc_upb_proto_reflection ALIAS CMakeProject_abc_upb_proto_reflection)

# cc_library(@upb_proto_library_test_repo//:x)
add_library(CMakeProject_x)
set_property(TARGET CMakeProject_x PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x PUBLIC
        "CMakeProject::abc_upb_proto"
        "CMakeProject::abc_upb_proto_reflection"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_x PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_x PUBLIC cxx_std_17)
target_sources(CMakeProject_x PRIVATE
        "${PROJECT_SOURCE_DIR}/x.cc")
add_library(CMakeProject::x ALIAS CMakeProject_x)
