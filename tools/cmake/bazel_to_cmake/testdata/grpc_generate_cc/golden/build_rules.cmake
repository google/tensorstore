find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# proto_library(@grpc_generate_cc_test_repo//:c_proto)
add_library(CMakeProject_c_proto INTERFACE)
target_sources(CMakeProject_c_proto INTERFACE
        "${TEST_DIRECTORY}/c.proto")
target_include_directories(CMakeProject_c_proto INTERFACE
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::c_proto ALIAS CMakeProject_c_proto)

# @grpc_generate_cc_test_repo//:c_proto__minitable_library
add_library(CMakeProject_c_proto__minitable_library)
set_property(TARGET CMakeProject_c_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__minitable_library PUBLIC
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_c_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto__minitable_library ALIAS CMakeProject_c_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_c_proto__minitable_library
    PROTO_TARGET CMakeProject_c_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/bdf6942a
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @grpc_generate_cc_test_repo//:c_proto__upb_library
add_library(CMakeProject_c_proto__upb_library)
set_property(TARGET CMakeProject_c_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upb_library PUBLIC
        "CMakeProject::c_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_c_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto__upb_library ALIAS CMakeProject_c_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_c_proto__upb_library
    PROTO_TARGET CMakeProject_c_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/bdf6942a
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# @grpc_generate_cc_test_repo//:cc__grpc_codegen
add_custom_target(CMakeProject_cc__grpc_codegen)

btc_protobuf(
    TARGET CMakeProject_cc__grpc_codegen
    PROTO_TARGET CMakeProject_c_proto
    LANGUAGE grpc
    GENERATE_EXTENSIONS ".grpc.pb.h" ".grpc.pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    PLUGIN protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    PLUGIN_OPTIONS "services_namespace=grpc_gen"
    DEPENDENCIES "CMakeProject_c_proto__upb_library" "gRPC::grpc_cpp_plugin" "protobuf::protoc"
)


# cc_library(@grpc_generate_cc_test_repo//:cc_grpc)
add_library(CMakeProject_cc_grpc)
set_property(TARGET CMakeProject_cc_grpc PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_cc_grpc PUBLIC
        "Threads::Threads"
        "gRPC::gRPC_codegen"
        "m")
target_include_directories(CMakeProject_cc_grpc PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_cc_grpc PUBLIC cxx_std_17)
add_dependencies(CMakeProject_cc_grpc "CMakeProject_cc__grpc_codegen")
target_sources(CMakeProject_cc_grpc PRIVATE
        "_cmake_binary_dir_/c.grpc.pb.cc")
add_library(CMakeProject::cc_grpc ALIAS CMakeProject_cc_grpc)

# cc_library(@grpc_generate_cc_test_repo//:a)
add_library(CMakeProject_a)
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "CMakeProject::cc_grpc"
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
add_library(CMakeProject::a ALIAS CMakeProject_a)

# @grpc_generate_cc_test_repo//:c_proto__cpp_library
add_library(CMakeProject_c_proto__cpp_library)
set_property(TARGET CMakeProject_c_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_c_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto__cpp_library ALIAS CMakeProject_c_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_c_proto__cpp_library
    PROTO_TARGET CMakeProject_c_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/bdf6942a
    DEPENDENCIES "protobuf::protoc"
)

# @grpc_generate_cc_test_repo//:c_proto__upbdefs_library
add_library(CMakeProject_c_proto__upbdefs_library)
set_property(TARGET CMakeProject_c_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upbdefs_library PUBLIC
        "CMakeProject::c_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port")
target_compile_features(CMakeProject_c_proto__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto__upbdefs_library ALIAS CMakeProject_c_proto__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_c_proto__upbdefs_library
    PROTO_TARGET CMakeProject_c_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/bdf6942a
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)
