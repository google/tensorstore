find_package(Protobuf REQUIRED)

# proto_library(@upb_proto_library_test_repo//:a_proto)
add_library(CMakeProject_a_proto INTERFACE)
target_sources(CMakeProject_a_proto INTERFACE
        "${PROJECT_SOURCE_DIR}/a.proto")
target_include_directories(CMakeProject_a_proto INTERFACE
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::a_proto ALIAS CMakeProject_a_proto)

# @upb_proto_library_test_repo//:a_proto__cpp_library
add_library(CMakeProject_a_proto__cpp_library)
set_property(TARGET CMakeProject_a_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_a_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::a_proto__cpp_library ALIAS CMakeProject_a_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_a_proto__cpp_library
    PROTO_TARGET CMakeProject_a_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/92f46a52
    DEPENDENCIES "protobuf::protoc"
)

# cc_proto_library(@upb_proto_library_test_repo//:a_cc_proto)
add_library(CMakeProject_a_cc_proto INTERFACE)
target_link_libraries(CMakeProject_a_cc_proto INTERFACE
        "CMakeProject::a_proto__cpp_library")
target_compile_features(CMakeProject_a_cc_proto INTERFACE cxx_std_17)
add_library(CMakeProject::a_cc_proto ALIAS CMakeProject_a_cc_proto)

# @upb_proto_library_test_repo//:a_proto__minitable_library
add_library(CMakeProject_a_proto__minitable_library)
set_property(TARGET CMakeProject_a_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__minitable_library PUBLIC
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_a_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::a_proto__minitable_library ALIAS CMakeProject_a_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_a_proto__minitable_library
    PROTO_TARGET CMakeProject_a_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/92f46a52
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @upb_proto_library_test_repo//:a_proto__upb_library
add_library(CMakeProject_a_proto__upb_library)
set_property(TARGET CMakeProject_a_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__upb_library PUBLIC
        "CMakeProject::a_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_a_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::a_proto__upb_library ALIAS CMakeProject_a_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_a_proto__upb_library
    PROTO_TARGET CMakeProject_a_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/92f46a52
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# upb_c_proto_library(@upb_proto_library_test_repo//:a_upb_proto)
add_library(CMakeProject_a_upb_proto INTERFACE)
target_link_libraries(CMakeProject_a_upb_proto INTERFACE
        "CMakeProject::a_proto__upb_library")
target_compile_features(CMakeProject_a_upb_proto INTERFACE cxx_std_17)
add_library(CMakeProject::a_upb_proto ALIAS CMakeProject_a_upb_proto)

# @upb_proto_library_test_repo//:a_proto__upbdefs_library
add_library(CMakeProject_a_proto__upbdefs_library)
set_property(TARGET CMakeProject_a_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__upbdefs_library PUBLIC
        "CMakeProject::a_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port")
target_compile_features(CMakeProject_a_proto__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::a_proto__upbdefs_library ALIAS CMakeProject_a_proto__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_a_proto__upbdefs_library
    PROTO_TARGET CMakeProject_a_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/92f46a52
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)

# upb_proto_reflection_library(@upb_proto_library_test_repo//:a_upb_proto_reflection)
add_library(CMakeProject_a_upb_proto_reflection INTERFACE)
target_link_libraries(CMakeProject_a_upb_proto_reflection INTERFACE
        "CMakeProject::a_proto__upbdefs_library")
target_compile_features(CMakeProject_a_upb_proto_reflection INTERFACE cxx_std_17)
add_library(CMakeProject::a_upb_proto_reflection ALIAS CMakeProject_a_upb_proto_reflection)

# upb_c_proto_library(@upb_proto_library_test_repo//:abc_upb_proto)
add_library(CMakeProject_abc_upb_proto INTERFACE)
target_link_libraries(CMakeProject_abc_upb_proto INTERFACE
        "CMakeProject::abc_protos__upb_library")
target_compile_features(CMakeProject_abc_upb_proto INTERFACE cxx_std_17)
add_library(CMakeProject::abc_upb_proto ALIAS CMakeProject_abc_upb_proto)

# upb_proto_reflection_library(@upb_proto_library_test_repo//:abc_upb_proto_reflection)
add_library(CMakeProject_abc_upb_proto_reflection INTERFACE)
target_link_libraries(CMakeProject_abc_upb_proto_reflection INTERFACE
        "CMakeProject::abc_protos__upbdefs_library")
target_compile_features(CMakeProject_abc_upb_proto_reflection INTERFACE cxx_std_17)
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
