find_package(Protobuf REQUIRED)
find_package(upb REQUIRED)

# @bazel_test_repo//:c_proto
add_library(CMakeProject_c_proto INTERFACE)
target_sources(CMakeProject_c_proto INTERFACE
        "${TEST_DIRECTORY}/c.proto")
list(APPEND CMakeProject_c_proto_IMPORT_DIRS "${TEST_DIRECTORY}")
set_property(TARGET CMakeProject_c_proto PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMakeProject_c_proto_IMPORT_DIRS})

# @bazel_test_repo//:c_proto__upb_library
add_library(CMakeProject_c_proto__upb_library)
target_sources(CMakeProject_c_proto__upb_library PRIVATE
        "${TEST_DIRECTORY}/c.proto")
set_property(TARGET CMakeProject_c_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upb_library PUBLIC
        "upb::generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_c_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto__upb_library PUBLIC cxx_std_17)

btc_protobuf(
    TARGET CMakeProject_c_proto__upb_library
    IMPORT_TARGETS  CMakeProject_c_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc-gen-upb>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc-gen-upb"
)

# @bazel_test_repo//:c_upb_proto
add_library(CMakeProject_c_upb_proto INTERFACE)
target_link_libraries(CMakeProject_c_upb_proto INTERFACE
        "CMakeProject_c_proto__upb_library")
target_include_directories(CMakeProject_c_upb_proto INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_upb_proto INTERFACE cxx_std_17)
add_library(CMakeProject::c_upb_proto ALIAS CMakeProject_c_upb_proto)

# @bazel_test_repo//:c_proto__upbdefs_library
add_library(CMakeProject_c_proto__upbdefs_library)
target_sources(CMakeProject_c_proto__upbdefs_library PRIVATE
        "${TEST_DIRECTORY}/c.proto")
set_property(TARGET CMakeProject_c_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upbdefs_library PUBLIC
        "upb::generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_c_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto__upbdefs_library PUBLIC cxx_std_17)

btc_protobuf(
    TARGET CMakeProject_c_proto__upbdefs_library
    IMPORT_TARGETS  CMakeProject_c_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:upb::protoc-gen-upbdefs>
    DEPENDENCIES "protobuf::protoc" "upb::protoc-gen-upbdefs"
)

# @bazel_test_repo//:c_upb_proto_reflection
add_library(CMakeProject_c_upb_proto_reflection INTERFACE)
target_link_libraries(CMakeProject_c_upb_proto_reflection INTERFACE
        "CMakeProject_c_proto__upb_library"
        "CMakeProject_c_proto__upbdefs_library")
target_include_directories(CMakeProject_c_upb_proto_reflection INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_upb_proto_reflection INTERFACE cxx_std_17)
add_library(CMakeProject::c_upb_proto_reflection ALIAS CMakeProject_c_upb_proto_reflection)

# @bazel_test_repo//:a
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
