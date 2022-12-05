find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

add_custom_command(
  OUTPUT "_cmake_binary_dir_/c.grpc.pb.h" "_cmake_binary_dir_/c.grpc.pb.cc"
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
      -I "${PROJECT_SOURCE_DIR}"
      --plugin=protoc-gen-grpc="$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
      --grpc_out=services_namespace=grpc_gen:${PROJECT_BINARY_DIR}
      "${TEST_DIRECTORY}/c.proto"
  DEPENDS "${TEST_DIRECTORY}/c.proto" "gRPC::grpc_cpp_plugin" "protobuf::protoc"
  COMMENT "Running protoc (grpc) on c.proto"
  VERBATIM)
add_custom_target(CMakeProject_cc__grpc_codegen DEPENDS "_cmake_binary_dir_/c.grpc.pb.h" "_cmake_binary_dir_/c.grpc.pb.cc")


add_library(CMakeProject_cc_grpc)
target_sources(CMakeProject_cc_grpc PRIVATE
        "_cmake_binary_dir_/c.grpc.pb.cc")
set_property(TARGET CMakeProject_cc_grpc PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_cc_grpc PUBLIC
        "Threads::Threads"
        "gRPC::gRPC_codegen"
        "m")
target_include_directories(CMakeProject_cc_grpc PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_cc_grpc PUBLIC cxx_std_17)
add_dependencies(CMakeProject_cc_grpc "CMakeProject_cc__grpc_codegen")
add_library(CMakeProject::cc_grpc ALIAS CMakeProject_cc_grpc)


add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "CMakeProject::cc_grpc"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_library(CMakeProject::a ALIAS CMakeProject_a)
