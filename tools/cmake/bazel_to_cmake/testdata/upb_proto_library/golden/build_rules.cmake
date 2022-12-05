find_package(Protobuf REQUIRED)
find_package(upb REQUIRED)

add_custom_command(
  OUTPUT "_cmake_binary_dir_/c.upbdefs.h" "_cmake_binary_dir_/c.upbdefs.cc"
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
      -I "${PROJECT_SOURCE_DIR}"
      --plugin=protoc-gen-upbdefs="$<TARGET_FILE:upb::protoc-gen-upbdefs>"
      --upbdefs_out=${PROJECT_BINARY_DIR}
      "${TEST_DIRECTORY}/c.proto"
  DEPENDS "${TEST_DIRECTORY}/c.proto" "protobuf::protoc" "upb::protoc-gen-upbdefs"
  COMMENT "Running protoc (upbdefs) on c.proto"
  VERBATIM)
add_custom_target(CMakeProject_c.proto__upbdefs_protoc DEPENDS "_cmake_binary_dir_/c.upbdefs.h" "_cmake_binary_dir_/c.upbdefs.cc")


add_library(CMakeProject_c.proto__upbdefs_proto)
target_sources(CMakeProject_c.proto__upbdefs_proto PRIVATE
        "_cmake_binary_dir_/c.upbdefs.cc")
set_property(TARGET CMakeProject_c.proto__upbdefs_proto PROPERTY LINKER_LANGUAGE "CXX")
target_include_directories(CMakeProject_c.proto__upbdefs_proto PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_c.proto__upbdefs_proto PUBLIC cxx_std_17)
add_library(CMakeProject::c.proto__upbdefs_proto ALIAS CMakeProject_c.proto__upbdefs_proto)
add_dependencies(CMakeProject_c.proto__upbdefs_proto CMakeProject_c.proto__upbdefs_protoc)


add_library(CMakeProject_c_upb_proto INTERFACE)
target_link_libraries(CMakeProject_c_upb_proto INTERFACE
        "CMakeProject::c.proto__upbdefs_proto")
target_include_directories(CMakeProject_c_upb_proto INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_c_upb_proto INTERFACE cxx_std_17)
add_library(CMakeProject::c_upb_proto ALIAS CMakeProject_c_upb_proto)

add_custom_command(
  OUTPUT "_cmake_binary_dir_/c.upb.h" "_cmake_binary_dir_/c.upb.cc"
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
      -I "${PROJECT_SOURCE_DIR}"
      --plugin=protoc-gen-upb="$<TARGET_FILE:protobuf::protoc-gen-upb>"
      --upb_out=${PROJECT_BINARY_DIR}
      "${TEST_DIRECTORY}/c.proto"
  DEPENDS "${TEST_DIRECTORY}/c.proto" "protobuf::protoc" "protobuf::protoc-gen-upb"
  COMMENT "Running protoc (upb) on c.proto"
  VERBATIM)
add_custom_target(CMakeProject_c.proto__upb_protoc DEPENDS "_cmake_binary_dir_/c.upb.h" "_cmake_binary_dir_/c.upb.cc")


add_library(CMakeProject_c.proto__upb_proto)
target_sources(CMakeProject_c.proto__upb_proto PRIVATE
        "_cmake_binary_dir_/c.upb.cc")
set_property(TARGET CMakeProject_c.proto__upb_proto PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c.proto__upb_proto PUBLIC
        "upb::generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_include_directories(CMakeProject_c.proto__upb_proto PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_c.proto__upb_proto PUBLIC cxx_std_17)
add_library(CMakeProject::c.proto__upb_proto ALIAS CMakeProject_c.proto__upb_proto)
add_dependencies(CMakeProject_c.proto__upb_proto CMakeProject_c.proto__upb_protoc)


add_library(CMakeProject_c_upb_proto_reflection INTERFACE)
target_link_libraries(CMakeProject_c_upb_proto_reflection INTERFACE
        "CMakeProject::c.proto__upb_proto")
target_include_directories(CMakeProject_c_upb_proto_reflection INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_c_upb_proto_reflection INTERFACE cxx_std_17)
add_library(CMakeProject::c_upb_proto_reflection ALIAS CMakeProject_c_upb_proto_reflection)


add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "CMakeProject::c_upb_proto"
        "CMakeProject::c_upb_proto_reflection"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_library(CMakeProject::a ALIAS CMakeProject_a)
