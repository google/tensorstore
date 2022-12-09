find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

add_custom_target(CMakeProject_c_proto)
list(APPEND CMakeProject_c_proto_INCLUDES ${TEST_DIRECTORY})

set_target_properties(CMakeProject_c_proto PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMakeProject_c_proto_INCLUDES}")

add_custom_command(
  OUTPUT "_cmake_binary_dir_/c.grpc.pb.h" "_cmake_binary_dir_/c.grpc.pb.cc"
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
       --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
       --grpc_out=services_namespace=grpc_gen:_cmake_binary_dir_
      "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
      "${TEST_DIRECTORY}/c.proto"
  DEPENDS "${TEST_DIRECTORY}/c.proto" "CMakeProject_c_proto" "gRPC::grpc_cpp_plugin" "protobuf::protoc"
  COMMENT "Running protoc (grpc) on c.proto"
  COMMAND_EXPAND_LISTS
  VERBATIM)
add_custom_target(CMakeProject_cc__grpc_codegen DEPENDS "_cmake_binary_dir_/c.grpc.pb.h" "_cmake_binary_dir_/c.grpc.pb.cc")
set_target_properties(CMakeProject_cc__grpc_codegen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES _cmake_binary_dir_)


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
