find_package(Protobuf REQUIRED)

add_custom_command(
  OUTPUT "_cmake_binary_dir_/a.h"
  DEPENDS "${TEST_DIRECTORY}/x.h"
  COMMAND cp x.h _cmake_binary_dir_/a.h
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(CMakeProject_h_file DEPENDS "_cmake_binary_dir_/a.h")


add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a "CMakeProject_h_file")
add_library(CMakeProject::a ALIAS CMakeProject_a)

add_custom_target(CMakeProject_c_proto)
list(APPEND CMakeProject_c_proto_INCLUDES ${TEST_DIRECTORY})

set_target_properties(CMakeProject_c_proto PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMakeProject_c_proto_INCLUDES}")

add_custom_command(
  OUTPUT "_cmake_binary_dir_/c.pb.h" "_cmake_binary_dir_/c.pb.cc"
  COMMAND protobuf::protoc
  ARGS --experimental_allow_proto3_optional
       --cpp_out=:_cmake_binary_dir_
      "-I$<JOIN:$<TARGET_PROPERTY:CMakeProject_c_proto,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
      "${TEST_DIRECTORY}/c.proto"
  DEPENDS "${TEST_DIRECTORY}/c.proto" "CMakeProject_c_proto" "protobuf::protoc"
  COMMENT "Running protoc (cpp) on c.proto"
  COMMAND_EXPAND_LISTS
  VERBATIM)
add_custom_target(CMakeProject_c.proto__cpp_protoc DEPENDS "_cmake_binary_dir_/c.pb.h" "_cmake_binary_dir_/c.pb.cc")
set_target_properties(CMakeProject_c.proto__cpp_protoc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES _cmake_binary_dir_)


add_library(CMakeProject_c_proto__cpp_library)
target_sources(CMakeProject_c_proto__cpp_library PRIVATE
        "_cmake_binary_dir_/c.pb.cc")
set_property(TARGET CMakeProject_c_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_c_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_c_proto__cpp_library PUBLIC cxx_std_17)
add_dependencies(CMakeProject_c_proto__cpp_library "CMakeProject_c.proto__cpp_protoc")
add_library(CMakeProject::c_proto__cpp_library ALIAS CMakeProject_c_proto__cpp_library)
target_include_directories(CMakeProject_c_proto__cpp_library PUBLIC
         $<BUILD_INTERFACE:$<TARGET_PROPERTY:CMakeProject_c.proto__cpp_protoc,INTERFACE_INCLUDE_DIRECTORIES>>)


add_library(CMakeProject_c_proto_cc INTERFACE)
target_link_libraries(CMakeProject_c_proto_cc INTERFACE
        "CMakeProject::c_proto__cpp_library"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_c_proto_cc INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_c_proto_cc INTERFACE cxx_std_17)
add_library(CMakeProject::c_proto_cc ALIAS CMakeProject_c_proto_cc)


add_executable(CMakeProject_a_test "")
add_executable(CMakeProject::a_test ALIAS CMakeProject_a_test)
target_sources(CMakeProject_a_test PRIVATE
        "${TEST_DIRECTORY}/a.cc")
target_link_libraries(CMakeProject_a_test PUBLIC
        "CMakeProject::c_proto_cc"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_test PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_a_test PUBLIC cxx_std_17)
add_test(NAME CMakeProject_a_test COMMAND CMakeProject_a_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
