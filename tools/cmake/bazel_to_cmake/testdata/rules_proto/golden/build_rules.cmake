find_package(Protobuf REQUIRED)

# @rules_proto_test_repo//:a_proto
add_library(CMakeProject_a_proto INTERFACE)
target_sources(CMakeProject_a_proto INTERFACE
        "${TEST_DIRECTORY}/a.proto")
target_include_directories(CMakeProject_a_proto INTERFACE
       "${TEST_DIRECTORY}")
add_library(CMakeProject::a_proto ALIAS CMakeProject_a_proto)

# @rules_proto_test_repo//:b_proto
add_library(CMakeProject_b_proto INTERFACE)
target_sources(CMakeProject_b_proto INTERFACE
        "${TEST_DIRECTORY}/b.proto")
target_include_directories(CMakeProject_b_proto INTERFACE
       "${TEST_DIRECTORY}")
add_library(CMakeProject::b_proto ALIAS CMakeProject_b_proto)

# @rules_proto_test_repo//:abc_proto
add_library(CMakeProject_abc_proto INTERFACE)
target_sources(CMakeProject_abc_proto INTERFACE
        "${TEST_DIRECTORY}/c.proto")
target_link_libraries(CMakeProject_abc_proto INTERFACE
        "CMakeProject::a_proto"
        "CMakeProject::b_proto")
target_include_directories(CMakeProject_abc_proto INTERFACE
       "${TEST_DIRECTORY}")
add_library(CMakeProject::abc_proto ALIAS CMakeProject_abc_proto)

# @rules_proto_test_repo//:a_proto__cpp_library
add_library(CMakeProject_a_proto__cpp_library)
set_property(TARGET CMakeProject_a_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_a_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::a_proto__cpp_library ALIAS CMakeProject_a_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_a_proto__cpp_library
    PROTO_TARGET CMakeProject_a_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    DEPENDENCIES "protobuf::protoc"
)

# @rules_proto_test_repo//:b_proto__cpp_library
add_library(CMakeProject_b_proto__cpp_library)
set_property(TARGET CMakeProject_b_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_b_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_b_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::b_proto__cpp_library ALIAS CMakeProject_b_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_b_proto__cpp_library
    PROTO_TARGET CMakeProject_b_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    DEPENDENCIES "protobuf::protoc"
)

# @rules_proto_test_repo//:abc_proto__cpp_library
add_library(CMakeProject_abc_proto__cpp_library)
set_property(TARGET CMakeProject_abc_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abc_proto__cpp_library PUBLIC
        "CMakeProject::a_proto__cpp_library"
        "CMakeProject::b_proto__cpp_library"
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_abc_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::abc_proto__cpp_library ALIAS CMakeProject_abc_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_abc_proto__cpp_library
    PROTO_TARGET CMakeProject_abc_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    DEPENDENCIES "protobuf::protoc"
)

# @rules_proto_test_repo//:abc_protos_cc
add_library(CMakeProject_abc_protos_cc INTERFACE)
target_link_libraries(CMakeProject_abc_protos_cc INTERFACE
        "CMakeProject::abc_proto__cpp_library")
target_include_directories(CMakeProject_abc_protos_cc INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_abc_protos_cc INTERFACE cxx_std_17)
add_library(CMakeProject::abc_protos_cc ALIAS CMakeProject_abc_protos_cc)

# @rules_proto_test_repo//:x_proto
add_library(CMakeProject_x_proto INTERFACE)
target_sources(CMakeProject_x_proto INTERFACE
        "${TEST_DIRECTORY}/x.proto")
target_link_libraries(CMakeProject_x_proto INTERFACE
        "Protobuf_any_proto")
target_include_directories(CMakeProject_x_proto INTERFACE
       ${Protobuf_IMPORT_DIRS}
       "${TEST_DIRECTORY}")
add_library(CMakeProject::x_proto ALIAS CMakeProject_x_proto)

# @rules_proto_test_repo//:x_proto__cpp_library
add_library(CMakeProject_x_proto__cpp_library)
set_property(TARGET CMakeProject_x_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_include_directories(CMakeProject_x_proto__cpp_library PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_x_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::x_proto__cpp_library ALIAS CMakeProject_x_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_x_proto__cpp_library
    PROTO_TARGET CMakeProject_x_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}
    DEPENDENCIES "protobuf::protoc"
)

# @rules_proto_test_repo//:x_proto_cc
add_library(CMakeProject_x_proto_cc INTERFACE)
target_link_libraries(CMakeProject_x_proto_cc INTERFACE
        "CMakeProject::x_proto__cpp_library")
target_include_directories(CMakeProject_x_proto_cc INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_x_proto_cc INTERFACE cxx_std_17)
add_library(CMakeProject::x_proto_cc ALIAS CMakeProject_x_proto_cc)
