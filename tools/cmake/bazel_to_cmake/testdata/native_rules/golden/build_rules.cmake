find_package(Protobuf REQUIRED)

# cc_binary(@native_rules_test_repo//:bb)
add_executable(CMakeProject_bb "")
add_executable(CMakeProject::bb ALIAS CMakeProject_bb)
target_compile_definitions(CMakeProject_bb PUBLIC "BUILD_LINUX")
target_link_libraries(CMakeProject_bb PUBLIC
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_bb PUBLIC cxx_std_17)
target_sources(CMakeProject_bb PRIVATE
        "${TEST_DIRECTORY}/a.cc")

# filegroup(@native_rules_test_repo//:subdir_z)
add_library(CMakeProject_subdir_z INTERFACE)
target_sources(CMakeProject_subdir_z INTERFACE
    "${TEST_DIRECTORY}/subdir/z.proto")
set_property(TARGET CMakeProject_subdir_z PROPERTY INTERFACE_IMPORTS
    "${PROJECT_SOURCE_DIR}")

# genrule(@native_rules_test_repo//:h_file)
add_custom_command(
  OUTPUT
    "_cmake_binary_dir_/a.h"
  DEPENDS
    "${TEST_DIRECTORY}/x.h"
    "CMakeProject::bb"
  COMMAND $<TARGET_FILE:CMakeProject_bb> ./a .  -I_cmake_binary_dir_/foo -Isubdir/../..  "x.h" "_cmake_binary_dir_/a.h"
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(genrule__CMakeProject_h_file DEPENDS
    "_cmake_binary_dir_/a.h")
add_library(CMakeProject_h_file INTERFACE)
target_sources(CMakeProject_h_file INTERFACE
    "_cmake_binary_dir_/a.h")
set_property(TARGET CMakeProject_h_file PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "${PROJECT_BINARY_DIR}")
add_dependencies(CMakeProject_h_file genrule__CMakeProject_h_file)

# cc_library(@native_rules_test_repo//:a)
add_library(CMakeProject_a)
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_compile_definitions(CMakeProject_a PUBLIC "BUILD_LINUX")
target_link_libraries(CMakeProject_a PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_dependencies(CMakeProject_a "CMakeProject_h_file")
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
add_library(CMakeProject::a ALIAS CMakeProject_a)

# alias(@native_rules_test_repo//:a_alias)
add_library(CMakeProject_a_alias ALIAS CMakeProject_a)
add_library(CMakeProject::a_alias ALIAS CMakeProject_a)

# proto_library(@native_rules_test_repo//:c_proto)
add_library(CMakeProject_c_proto INTERFACE)
target_sources(CMakeProject_c_proto INTERFACE
        "${TEST_DIRECTORY}/c.proto")
target_link_libraries(CMakeProject_c_proto INTERFACE
        "Protobuf_timestamp_proto")
target_include_directories(CMakeProject_c_proto INTERFACE
       ${Protobuf_IMPORT_DIRS}
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::c_proto ALIAS CMakeProject_c_proto)

# @native_rules_test_repo//:c_proto__cpp_library
add_library(CMakeProject_c_proto__cpp_library)
set_property(TARGET CMakeProject_c_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__cpp_library PUBLIC
        "Protobuf_timestamp_proto__cpp_library"
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_c_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto__cpp_library ALIAS CMakeProject_c_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_c_proto__cpp_library
    PROTO_TARGET CMakeProject_c_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/de6f8ca9
    DEPENDENCIES "protobuf::protoc"
)

# cc_proto_library(@native_rules_test_repo//:c_proto_cc)
add_library(CMakeProject_c_proto_cc INTERFACE)
target_link_libraries(CMakeProject_c_proto_cc INTERFACE
        "CMakeProject::c_proto__cpp_library")
target_compile_features(CMakeProject_c_proto_cc INTERFACE cxx_std_17)
add_library(CMakeProject::c_proto_cc ALIAS CMakeProject_c_proto_cc)

# cc_test(@native_rules_test_repo//:a_test)
add_executable(CMakeProject_a_test "")
add_executable(CMakeProject::a_test ALIAS CMakeProject_a_test)
target_compile_definitions(CMakeProject_a_test PUBLIC "BUILD_LINUX")
target_link_libraries(CMakeProject_a_test PUBLIC
        "CMakeProject::c_proto_cc"
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_a_test PUBLIC cxx_std_17)
target_sources(CMakeProject_a_test PRIVATE
        "${TEST_DIRECTORY}/a.cc")
add_test(NAME CMakeProject_a_test COMMAND CMakeProject_a_test WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# proto_library(@native_rules_test_repo//:c_proto_2)
add_library(CMakeProject_c_proto_2 INTERFACE)
target_sources(CMakeProject_c_proto_2 INTERFACE
        "${TEST_DIRECTORY}/c.proto")
target_include_directories(CMakeProject_c_proto_2 INTERFACE
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::c_proto_2 ALIAS CMakeProject_c_proto_2)

# @native_rules_test_repo//:c_proto_2__cpp_library
add_library(CMakeProject_c_proto_2__cpp_library)
set_property(TARGET CMakeProject_c_proto_2__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto_2__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_c_proto_2__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto_2__cpp_library ALIAS CMakeProject_c_proto_2__cpp_library)

btc_protobuf(
    TARGET CMakeProject_c_proto_2__cpp_library
    PROTO_TARGET CMakeProject_c_proto_2
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/c5077a10
    DEPENDENCIES "protobuf::protoc"
)

# @native_rules_test_repo//:c_proto_2__minitable_library
add_library(CMakeProject_c_proto_2__minitable_library)
set_property(TARGET CMakeProject_c_proto_2__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto_2__minitable_library PUBLIC
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_c_proto_2__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto_2__minitable_library ALIAS CMakeProject_c_proto_2__minitable_library)

btc_protobuf(
    TARGET CMakeProject_c_proto_2__minitable_library
    PROTO_TARGET CMakeProject_c_proto_2
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/c5077a10
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @native_rules_test_repo//:c_proto_2__upb_library
add_library(CMakeProject_c_proto_2__upb_library)
set_property(TARGET CMakeProject_c_proto_2__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto_2__upb_library PUBLIC
        "CMakeProject::c_proto_2__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_c_proto_2__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto_2__upb_library ALIAS CMakeProject_c_proto_2__upb_library)

btc_protobuf(
    TARGET CMakeProject_c_proto_2__upb_library
    PROTO_TARGET CMakeProject_c_proto_2
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/c5077a10
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# @native_rules_test_repo//:c_proto_2__upbdefs_library
add_library(CMakeProject_c_proto_2__upbdefs_library)
set_property(TARGET CMakeProject_c_proto_2__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto_2__upbdefs_library PUBLIC
        "CMakeProject::c_proto_2__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port")
target_compile_features(CMakeProject_c_proto_2__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto_2__upbdefs_library ALIAS CMakeProject_c_proto_2__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_c_proto_2__upbdefs_library
    PROTO_TARGET CMakeProject_c_proto_2
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/c5077a10
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)

# @native_rules_test_repo//:c_proto__minitable_library
add_library(CMakeProject_c_proto__minitable_library)
set_property(TARGET CMakeProject_c_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__minitable_library PUBLIC
        "Protobuf_timestamp_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_c_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto__minitable_library ALIAS CMakeProject_c_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_c_proto__minitable_library
    PROTO_TARGET CMakeProject_c_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/de6f8ca9
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @native_rules_test_repo//:c_proto__upb_library
add_library(CMakeProject_c_proto__upb_library)
set_property(TARGET CMakeProject_c_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upb_library PUBLIC
        "CMakeProject::c_proto__minitable_library"
        "Protobuf_timestamp_proto__upb_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_c_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto__upb_library ALIAS CMakeProject_c_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_c_proto__upb_library
    PROTO_TARGET CMakeProject_c_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/de6f8ca9
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# @native_rules_test_repo//:c_proto__upbdefs_library
add_library(CMakeProject_c_proto__upbdefs_library)
set_property(TARGET CMakeProject_c_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_c_proto__upbdefs_library PUBLIC
        "CMakeProject::c_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port"
        "Protobuf_timestamp_proto__upbdefs_library")
target_compile_features(CMakeProject_c_proto__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::c_proto__upbdefs_library ALIAS CMakeProject_c_proto__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_c_proto__upbdefs_library
    PROTO_TARGET CMakeProject_c_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/de6f8ca9
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)

# alias(@native_rules_test_repo//:c_proto_alias)
add_library(CMakeProject_c_proto_alias ALIAS CMakeProject_c_proto)
add_library(CMakeProject::c_proto_alias ALIAS CMakeProject_c_proto)

# alias(@native_rules_test_repo//:c_proto_cc_alias)
add_library(CMakeProject_c_proto_cc_alias ALIAS CMakeProject_c_proto_cc)
add_library(CMakeProject::c_proto_cc_alias ALIAS CMakeProject_c_proto_cc)

# cc_library(@native_rules_test_repo//:subdir_x)
add_library(CMakeProject_subdir_x)
set_property(TARGET CMakeProject_subdir_x PROPERTY LINKER_LANGUAGE "CXX")
target_compile_definitions(CMakeProject_subdir_x PUBLIC "BUILD_LINUX")
target_link_libraries(CMakeProject_subdir_x PUBLIC
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_subdir_x PUBLIC cxx_std_17)
target_sources(CMakeProject_subdir_x PRIVATE
        "${TEST_DIRECTORY}/subdir/x.cc")
add_library(CMakeProject::subdir_x ALIAS CMakeProject_subdir_x)

# genrule(@native_rules_test_repo//subdir:make_ycc)
add_custom_command(
  OUTPUT
    "_cmake_binary_dir_/subdir/y.cc"
  DEPENDS
    "CMakeProject::bb"
  COMMAND $<TARGET_FILE:CMakeProject_bb> "_cmake_binary_dir_/subdir/y.cc"
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(genrule__CMakeProject_subdir_make_ycc DEPENDS
    "_cmake_binary_dir_/subdir/y.cc")
add_library(CMakeProject_subdir_make_ycc INTERFACE)
target_sources(CMakeProject_subdir_make_ycc INTERFACE
    "_cmake_binary_dir_/subdir/y.cc")
set_property(TARGET CMakeProject_subdir_make_ycc PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "${PROJECT_BINARY_DIR}")
add_dependencies(CMakeProject_subdir_make_ycc genrule__CMakeProject_subdir_make_ycc)

# genrule(@native_rules_test_repo//subdir:make_y)
add_custom_command(
  OUTPUT
    "_cmake_binary_dir_/subdir/y.h"
  DEPENDS
    "CMakeProject::bb"
  COMMAND $<TARGET_FILE:CMakeProject_bb> "_cmake_binary_dir_/subdir/y.h"
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(genrule__CMakeProject_subdir_make_y DEPENDS
    "_cmake_binary_dir_/subdir/y.h")
add_library(CMakeProject_subdir_make_y INTERFACE)
target_sources(CMakeProject_subdir_make_y INTERFACE
    "_cmake_binary_dir_/subdir/y.h")
set_property(TARGET CMakeProject_subdir_make_y PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "${PROJECT_BINARY_DIR}")
add_dependencies(CMakeProject_subdir_make_y genrule__CMakeProject_subdir_make_y)

# cc_library(@native_rules_test_repo//subdir:y)
add_library(CMakeProject_subdir_y)
set_property(TARGET CMakeProject_subdir_y PROPERTY LINKER_LANGUAGE "CXX")
target_compile_definitions(CMakeProject_subdir_y PUBLIC "BUILD_LINUX")
target_link_libraries(CMakeProject_subdir_y PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_subdir_y PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_subdir_y PUBLIC cxx_std_17)
add_dependencies(CMakeProject_subdir_y "CMakeProject_subdir_make_y" "CMakeProject_subdir_make_ycc")
target_sources(CMakeProject_subdir_y PRIVATE
        "_cmake_binary_dir_/subdir/y.cc")
add_library(CMakeProject::subdir_y ALIAS CMakeProject_subdir_y)

# alias(@native_rules_test_repo//:y_alias)
add_library(CMakeProject_y_alias ALIAS CMakeProject_subdir_y)
add_library(CMakeProject::y_alias ALIAS CMakeProject_subdir_y)

# cc_library(@native_rules_test_repo//subdir:y_include_prefix)
add_library(CMakeProject_subdir_y_include_prefix)
set_property(TARGET CMakeProject_subdir_y_include_prefix PROPERTY LINKER_LANGUAGE "CXX")
target_compile_definitions(CMakeProject_subdir_y_include_prefix PUBLIC "BUILD_LINUX")
target_link_libraries(CMakeProject_subdir_y_include_prefix PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_subdir_y_include_prefix PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_subdir_y_include_prefix PUBLIC cxx_std_17)
add_dependencies(CMakeProject_subdir_y_include_prefix "CMakeProject_subdir_make_y" "CMakeProject_subdir_make_ycc")
target_sources(CMakeProject_subdir_y_include_prefix PRIVATE
        "_cmake_binary_dir_/subdir/y.cc")
add_library(CMakeProject::subdir_y_include_prefix ALIAS CMakeProject_subdir_y_include_prefix)

# cc_library(@native_rules_test_repo//subdir:y_includes)
add_library(CMakeProject_subdir_y_includes)
set_property(TARGET CMakeProject_subdir_y_includes PROPERTY LINKER_LANGUAGE "CXX")
target_compile_definitions(CMakeProject_subdir_y_includes PUBLIC "BUILD_LINUX")
target_link_libraries(CMakeProject_subdir_y_includes PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_subdir_y_includes PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/subdir>")
target_compile_features(CMakeProject_subdir_y_includes PUBLIC cxx_std_17)
add_dependencies(CMakeProject_subdir_y_includes "CMakeProject_subdir_make_y" "CMakeProject_subdir_make_ycc")
target_sources(CMakeProject_subdir_y_includes PRIVATE
        "_cmake_binary_dir_/subdir/y.cc")
add_library(CMakeProject::subdir_y_includes ALIAS CMakeProject_subdir_y_includes)

# cc_library(@native_rules_test_repo//subdir:y_strip_include_prefix)
add_library(CMakeProject_subdir_y_strip_include_prefix)
set_property(TARGET CMakeProject_subdir_y_strip_include_prefix PROPERTY LINKER_LANGUAGE "CXX")
target_compile_definitions(CMakeProject_subdir_y_strip_include_prefix PUBLIC "BUILD_LINUX")
target_link_libraries(CMakeProject_subdir_y_strip_include_prefix PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_subdir_y_strip_include_prefix PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/subdir>")
target_compile_features(CMakeProject_subdir_y_strip_include_prefix PUBLIC cxx_std_17)
add_dependencies(CMakeProject_subdir_y_strip_include_prefix "CMakeProject_subdir_make_y" "CMakeProject_subdir_make_ycc")
target_sources(CMakeProject_subdir_y_strip_include_prefix PRIVATE
        "_cmake_binary_dir_/subdir/y.cc")
add_library(CMakeProject::subdir_y_strip_include_prefix ALIAS CMakeProject_subdir_y_strip_include_prefix)

# proto_library(@native_rules_test_repo//subdir:z_proto)
add_library(CMakeProject_subdir_z_proto INTERFACE)
target_sources(CMakeProject_subdir_z_proto INTERFACE
        "${TEST_DIRECTORY}/subdir/z.proto")
target_include_directories(CMakeProject_subdir_z_proto INTERFACE
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::subdir_z_proto ALIAS CMakeProject_subdir_z_proto)

# @native_rules_test_repo//subdir:z_proto__cpp_library
add_library(CMakeProject_subdir_z_proto__cpp_library)
set_property(TARGET CMakeProject_subdir_z_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_z_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_subdir_z_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::subdir_z_proto__cpp_library ALIAS CMakeProject_subdir_z_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_subdir_z_proto__cpp_library
    PROTO_TARGET CMakeProject_subdir_z_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/fb9b71f5
    DEPENDENCIES "protobuf::protoc"
)

# @native_rules_test_repo//subdir:z_proto__minitable_library
add_library(CMakeProject_subdir_z_proto__minitable_library)
set_property(TARGET CMakeProject_subdir_z_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_z_proto__minitable_library PUBLIC
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_subdir_z_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::subdir_z_proto__minitable_library ALIAS CMakeProject_subdir_z_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_subdir_z_proto__minitable_library
    PROTO_TARGET CMakeProject_subdir_z_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/fb9b71f5
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @native_rules_test_repo//subdir:z_proto__upb_library
add_library(CMakeProject_subdir_z_proto__upb_library)
set_property(TARGET CMakeProject_subdir_z_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_z_proto__upb_library PUBLIC
        "CMakeProject::subdir_z_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_subdir_z_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::subdir_z_proto__upb_library ALIAS CMakeProject_subdir_z_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_subdir_z_proto__upb_library
    PROTO_TARGET CMakeProject_subdir_z_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/fb9b71f5
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# @native_rules_test_repo//subdir:z_proto__upbdefs_library
add_library(CMakeProject_subdir_z_proto__upbdefs_library)
set_property(TARGET CMakeProject_subdir_z_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_subdir_z_proto__upbdefs_library PUBLIC
        "CMakeProject::subdir_z_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port")
target_compile_features(CMakeProject_subdir_z_proto__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::subdir_z_proto__upbdefs_library ALIAS CMakeProject_subdir_z_proto__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_subdir_z_proto__upbdefs_library
    PROTO_TARGET CMakeProject_subdir_z_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/fb9b71f5
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDENCIES "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)
