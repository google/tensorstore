find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# @bazel_test_repo//:c_proto
add_library(CMakeProject_c_proto INTERFACE)
target_sources(CMakeProject_c_proto INTERFACE
        "${TEST_DIRECTORY}/c.proto")
list(APPEND CMakeProject_c_proto_IMPORT_DIRS "${TEST_DIRECTORY}")
set_property(TARGET CMakeProject_c_proto PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMakeProject_c_proto_IMPORT_DIRS})

# @bazel_test_repo//:cc__grpc_codegen
add_custom_target(CMakeProject_cc__grpc_codegen)
target_sources(CMakeProject_cc__grpc_codegen PRIVATE
        "${TEST_DIRECTORY}/c.proto")

btc_protobuf(
    TARGET CMakeProject_cc__grpc_codegen
    IMPORT_TARGETS  CMakeProject_c_proto
    LANGUAGE grpc
    PLUGIN protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    GENERATE_EXTENSIONS ".grpc.pb.h" ".grpc.pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PLUGIN_OPTIONS "services_namespace=grpc_gen"
    DEPENDENCIES "protobuf::protoc" "gRPC::grpc_cpp_plugin"
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
)

# @bazel_test_repo//:cc_grpc
add_library(CMakeProject_cc_grpc)
target_sources(CMakeProject_cc_grpc PRIVATE
        "_cmake_binary_dir_/c.grpc.pb.cc")
set_property(TARGET CMakeProject_cc_grpc PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_cc_grpc PUBLIC
        "Threads::Threads"
        "gRPC::gRPC_codegen"
        "m")
target_include_directories(CMakeProject_cc_grpc PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_cc_grpc PUBLIC cxx_std_17)
add_dependencies(CMakeProject_cc_grpc "CMakeProject_cc__grpc_codegen")
add_library(CMakeProject::cc_grpc ALIAS CMakeProject_cc_grpc)

# @bazel_test_repo//:a
add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "CMakeProject::cc_grpc"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_library(CMakeProject::a ALIAS CMakeProject_a)
