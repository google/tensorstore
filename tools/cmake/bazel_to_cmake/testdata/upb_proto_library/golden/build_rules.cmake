find_package(Protobuf REQUIRED)
find_package(upb REQUIRED)

# @upb_proto_library_test_repo//:c_proto
add_library(CMakeProject_c_proto INTERFACE)
target_sources(CMakeProject_c_proto INTERFACE
        "${TEST_DIRECTORY}/c.proto")
list(APPEND CMakeProject_c_proto_IMPORT_DIRS "${TEST_DIRECTORY}")
set_property(TARGET CMakeProject_c_proto PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMakeProject_c_proto_IMPORT_DIRS})

# @upb_proto_library_test_repo//:c_proto__upb_library
add_library(CMakeProject_c_proto__upb_library)
set_property(TARGET CMakeProject_c_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upb_library PUBLIC
        "upb::generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_c_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto__upb_library PUBLIC cxx_std_17)
target_sources(CMakeProject_c_proto__upb_library PRIVATE
        "${TEST_DIRECTORY}/c.proto")

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

# @upb_proto_library_test_repo//:c_upb_proto
add_library(CMakeProject_c_upb_proto INTERFACE)
target_link_libraries(CMakeProject_c_upb_proto INTERFACE
        "CMakeProject_c_proto__upb_library")
target_include_directories(CMakeProject_c_upb_proto INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_upb_proto INTERFACE cxx_std_17)
add_library(CMakeProject::c_upb_proto ALIAS CMakeProject_c_upb_proto)

# @upb_proto_library_test_repo//:c_proto__upbdefs_library
add_library(CMakeProject_c_proto__upbdefs_library)
set_property(TARGET CMakeProject_c_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upbdefs_library PUBLIC
        "upb::generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_c_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto__upbdefs_library PUBLIC cxx_std_17)
target_sources(CMakeProject_c_proto__upbdefs_library PRIVATE
        "${TEST_DIRECTORY}/c.proto")

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

# @upb_proto_library_test_repo//:c_upb_proto_reflection
add_library(CMakeProject_c_upb_proto_reflection INTERFACE)
target_link_libraries(CMakeProject_c_upb_proto_reflection INTERFACE
        "CMakeProject_c_proto__upb_library"
        "CMakeProject_c_proto__upbdefs_library")
target_include_directories(CMakeProject_c_upb_proto_reflection INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_upb_proto_reflection INTERFACE cxx_std_17)
add_library(CMakeProject::c_upb_proto_reflection ALIAS CMakeProject_c_upb_proto_reflection)

# @upb_proto_library_test_repo//:a
add_library(CMakeProject_a)
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
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
add_library(CMakeProject::a ALIAS CMakeProject_a)

# @upb_proto_library_test_repo//:d_proto
add_library(CMakeProject_d_proto INTERFACE)
target_sources(CMakeProject_d_proto INTERFACE
        "${TEST_DIRECTORY}/d.proto")
list(APPEND CMakeProject_d_proto_IMPORT_DIRS "${TEST_DIRECTORY}")
set_property(TARGET CMakeProject_d_proto PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMakeProject_d_proto_IMPORT_DIRS})

# @upb_proto_library_test_repo//:abc_protos
add_library(CMakeProject_abc_protos INTERFACE)
target_sources(CMakeProject_abc_protos INTERFACE
        )
btc_transitive_import_dirs(
    OUT_VAR CMakeProject_abc_protos_IMPORT_DIRS
    IMPORT_DIRS 
    IMPORT_TARGETS CMakeProject_c_proto CMakeProject_d_proto 
    IMPORT_VARS 
)
set_property(TARGET CMakeProject_abc_protos PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CMakeProject_abc_protos_IMPORT_DIRS})

# @upb_proto_library_test_repo//:c_proto__cpp_library
add_library(CMakeProject_c_proto__cpp_library)
set_property(TARGET CMakeProject_c_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_c_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_c_proto__cpp_library PUBLIC cxx_std_17)
target_sources(CMakeProject_c_proto__cpp_library PRIVATE
        "${TEST_DIRECTORY}/c.proto")

btc_protobuf(
    TARGET CMakeProject_c_proto__cpp_library
    IMPORT_TARGETS  CMakeProject_c_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    DEPENDENCIES "protobuf::protoc"
)

# @upb_proto_library_test_repo//:d_proto__cpp_library
add_library(CMakeProject_d_proto__cpp_library)
set_property(TARGET CMakeProject_d_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_d_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_d_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_d_proto__cpp_library PUBLIC cxx_std_17)
target_sources(CMakeProject_d_proto__cpp_library PRIVATE
        "${TEST_DIRECTORY}/d.proto")

btc_protobuf(
    TARGET CMakeProject_d_proto__cpp_library
    IMPORT_TARGETS  CMakeProject_d_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    DEPENDENCIES "protobuf::protoc"
)

# @upb_proto_library_test_repo//:abc_protos__cpp_library
add_library(CMakeProject_abc_protos__cpp_library INTERFACE)
target_link_libraries(CMakeProject_abc_protos__cpp_library INTERFACE
        "CMakeProject_c_proto__cpp_library"
        "CMakeProject_d_proto__cpp_library"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_abc_protos__cpp_library INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_protos__cpp_library INTERFACE cxx_std_17)

# @upb_proto_library_test_repo//:abc_protos_cc
add_library(CMakeProject_abc_protos_cc INTERFACE)
target_link_libraries(CMakeProject_abc_protos_cc INTERFACE
        "CMakeProject_abc_protos__cpp_library")
target_include_directories(CMakeProject_abc_protos_cc INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_protos_cc INTERFACE cxx_std_17)
add_library(CMakeProject::abc_protos_cc ALIAS CMakeProject_abc_protos_cc)

# @upb_proto_library_test_repo//:d_proto__upb_library
add_library(CMakeProject_d_proto__upb_library)
set_property(TARGET CMakeProject_d_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_d_proto__upb_library PUBLIC
        "upb::generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_d_proto__upb_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_d_proto__upb_library PUBLIC cxx_std_17)
target_sources(CMakeProject_d_proto__upb_library PRIVATE
        "${TEST_DIRECTORY}/d.proto")

btc_protobuf(
    TARGET CMakeProject_d_proto__upb_library
    IMPORT_TARGETS  CMakeProject_d_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc-gen-upb>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc-gen-upb"
)

# @upb_proto_library_test_repo//:abc_protos__upb_library
add_library(CMakeProject_abc_protos__upb_library INTERFACE)
target_link_libraries(CMakeProject_abc_protos__upb_library INTERFACE
        "CMakeProject_c_proto__upb_library"
        "CMakeProject_d_proto__upb_library"
        "upb::generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_abc_protos__upb_library INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_protos__upb_library INTERFACE cxx_std_17)

# @upb_proto_library_test_repo//:abc_protos_upb
add_library(CMakeProject_abc_protos_upb INTERFACE)
target_link_libraries(CMakeProject_abc_protos_upb INTERFACE
        "CMakeProject_abc_protos__upb_library")
target_include_directories(CMakeProject_abc_protos_upb INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_protos_upb INTERFACE cxx_std_17)
add_library(CMakeProject::abc_protos_upb ALIAS CMakeProject_abc_protos_upb)

# @upb_proto_library_test_repo//:d_proto__upbdefs_library
add_library(CMakeProject_d_proto__upbdefs_library)
set_property(TARGET CMakeProject_d_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_d_proto__upbdefs_library PUBLIC
        "upb::generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_d_proto__upbdefs_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_d_proto__upbdefs_library PUBLIC cxx_std_17)
target_sources(CMakeProject_d_proto__upbdefs_library PRIVATE
        "${TEST_DIRECTORY}/d.proto")

btc_protobuf(
    TARGET CMakeProject_d_proto__upbdefs_library
    IMPORT_TARGETS  CMakeProject_d_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:upb::protoc-gen-upbdefs>
    DEPENDENCIES "protobuf::protoc" "upb::protoc-gen-upbdefs"
)

# @upb_proto_library_test_repo//:abc_protos__upbdefs_library
add_library(CMakeProject_abc_protos__upbdefs_library INTERFACE)
target_link_libraries(CMakeProject_abc_protos__upbdefs_library INTERFACE
        "CMakeProject_c_proto__upbdefs_library"
        "CMakeProject_d_proto__upbdefs_library"
        "upb::generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "upb::port")
target_include_directories(CMakeProject_abc_protos__upbdefs_library INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_protos__upbdefs_library INTERFACE cxx_std_17)

# @upb_proto_library_test_repo//:abc_protos_upbdefs
add_library(CMakeProject_abc_protos_upbdefs INTERFACE)
target_link_libraries(CMakeProject_abc_protos_upbdefs INTERFACE
        "CMakeProject_abc_protos__upb_library"
        "CMakeProject_abc_protos__upbdefs_library")
target_include_directories(CMakeProject_abc_protos_upbdefs INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_protos_upbdefs INTERFACE cxx_std_17)
add_library(CMakeProject::abc_protos_upbdefs ALIAS CMakeProject_abc_protos_upbdefs)