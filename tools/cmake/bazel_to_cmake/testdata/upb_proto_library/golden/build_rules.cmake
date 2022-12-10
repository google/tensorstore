find_package(Protobuf REQUIRED)
find_package(upb REQUIRED)

add_custom_target(CMakeProject_c_proto)
list(APPEND CMakeProject_c_proto_INCLUDES ${TEST_DIRECTORY})

set_target_properties(CMakeProject_c_proto PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMakeProject_c_proto_INCLUDES}")

add_custom_command(
  OUTPUT "_cmake_binary_dir_/c.upb.h" "_cmake_binary_dir_/c.upb.c"
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
       --plugin=protoc-gen-upb=$<TARGET_FILE:protobuf::protoc-gen-upb>
       --upb_out=:_cmake_binary_dir_
      "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
      "${TEST_DIRECTORY}/c.proto"
  DEPENDS "${TEST_DIRECTORY}/c.proto" "CMakeProject_c_proto" "protobuf::protoc" "protobuf::protoc-gen-upb"
  COMMENT "Running protoc (upb) on c.proto"
  COMMAND_EXPAND_LISTS
  VERBATIM)
add_custom_target(CMakeProject_c.proto__upb_protoc DEPENDS "_cmake_binary_dir_/c.upb.h" "_cmake_binary_dir_/c.upb.c")
set_target_properties(CMakeProject_c.proto__upb_protoc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES _cmake_binary_dir_)


add_library(CMakeProject_c_proto__upb_library)
target_sources(CMakeProject_c_proto__upb_library PRIVATE
        "_cmake_binary_dir_/c.upb.c")
set_property(TARGET CMakeProject_c_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upb_library PUBLIC
        "upb::generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_c_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto__upb_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_c_proto__upb_library "CMakeProject_c.proto__upb_protoc")
add_library(CMakeProject::c_proto__upb_library ALIAS CMakeProject_c_proto__upb_library)
target_include_directories(CMakeProject_c_proto__upb_library PUBLIC
         $<BUILD_INTERFACE:$<TARGET_PROPERTY:CMakeProject_c.proto__upb_protoc,INTERFACE_INCLUDE_DIRECTORIES>>)


add_library(CMakeProject_c_upb_proto INTERFACE)
target_link_libraries(CMakeProject_c_upb_proto INTERFACE
        "CMakeProject::c_proto__upb_library")
target_include_directories(CMakeProject_c_upb_proto INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_upb_proto INTERFACE cxx_std_17)
add_library(CMakeProject::c_upb_proto ALIAS CMakeProject_c_upb_proto)

add_custom_command(
  OUTPUT "_cmake_binary_dir_/c.upbdefs.h" "_cmake_binary_dir_/c.upbdefs.c"
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
       --plugin=protoc-gen-upbdefs=$<TARGET_FILE:upb::protoc-gen-upbdefs>
       --upbdefs_out=:_cmake_binary_dir_
      "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
      "${TEST_DIRECTORY}/c.proto"
  DEPENDS "${TEST_DIRECTORY}/c.proto" "CMakeProject_c_proto" "protobuf::protoc" "upb::protoc-gen-upbdefs"
  COMMENT "Running protoc (upbdefs) on c.proto"
  COMMAND_EXPAND_LISTS
  VERBATIM)
add_custom_target(CMakeProject_c.proto__upbdefs_protoc DEPENDS "_cmake_binary_dir_/c.upbdefs.h" "_cmake_binary_dir_/c.upbdefs.c")
set_target_properties(CMakeProject_c.proto__upbdefs_protoc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES _cmake_binary_dir_)


add_library(CMakeProject_c_proto__upbdefs_library)
target_sources(CMakeProject_c_proto__upbdefs_library PRIVATE
        "_cmake_binary_dir_/c.upbdefs.c")
set_property(TARGET CMakeProject_c_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upbdefs_library PUBLIC
        "upb::generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_c_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto__upbdefs_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_c_proto__upbdefs_library "CMakeProject_c.proto__upbdefs_protoc")
add_library(CMakeProject::c_proto__upbdefs_library ALIAS CMakeProject_c_proto__upbdefs_library)
target_include_directories(CMakeProject_c_proto__upbdefs_library PUBLIC
         $<BUILD_INTERFACE:$<TARGET_PROPERTY:CMakeProject_c.proto__upbdefs_protoc,INTERFACE_INCLUDE_DIRECTORIES>>)


add_library(CMakeProject_c_upb_proto_reflection INTERFACE)
target_link_libraries(CMakeProject_c_upb_proto_reflection INTERFACE
        "CMakeProject::c_proto__upb_library"
        "CMakeProject::c_proto__upbdefs_library")
target_include_directories(CMakeProject_c_upb_proto_reflection INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
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
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_library(CMakeProject::a ALIAS CMakeProject_a)
