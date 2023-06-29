
# cc_library(@upb_bootstrap_compiler_test_repo//:b)
add_library(CMakeProject_b)
set_property(TARGET CMakeProject_b PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b PUBLIC
        "CMakeProject::c_upb"
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_b PUBLIC cxx_std_17)
target_sources(CMakeProject_b PRIVATE
        "${TEST_DIRECTORY}/a.cc")
add_library(CMakeProject::b ALIAS CMakeProject_b)

# cc_binary(@upb_bootstrap_compiler_test_repo//:a)
add_executable(CMakeProject_a "")
add_executable(CMakeProject::a ALIAS CMakeProject_a)
target_link_libraries(CMakeProject_a PUBLIC
        "CMakeProject::b"
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")

# cc_library(@upb_bootstrap_compiler_test_repo//:b_stage0)
add_library(CMakeProject_b_stage0)
set_property(TARGET CMakeProject_b_stage0 PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_stage0 PUBLIC
        "CMakeProject::c_upb_stage0"
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_b_stage0 PUBLIC cxx_std_17)
target_sources(CMakeProject_b_stage0 PRIVATE
        "${TEST_DIRECTORY}/a.cc")
add_library(CMakeProject::b_stage0 ALIAS CMakeProject_b_stage0)

# cc_binary(@upb_bootstrap_compiler_test_repo//:a_stage0)
add_executable(CMakeProject_a_stage0 "")
add_executable(CMakeProject::a_stage0 ALIAS CMakeProject_a_stage0)
target_link_libraries(CMakeProject_a_stage0 PUBLIC
        "CMakeProject::b_stage0"
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_a_stage0 PUBLIC cxx_std_17)
target_sources(CMakeProject_a_stage0 PRIVATE
        "${TEST_DIRECTORY}/a.cc")

# cc_library(@upb_bootstrap_compiler_test_repo//:b_stage1)
add_library(CMakeProject_b_stage1)
set_property(TARGET CMakeProject_b_stage1 PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_stage1 PUBLIC
        "CMakeProject::c_upb_stage1"
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_b_stage1 PUBLIC cxx_std_17)
target_sources(CMakeProject_b_stage1 PRIVATE
        "${TEST_DIRECTORY}/a.cc")
add_library(CMakeProject::b_stage1 ALIAS CMakeProject_b_stage1)

# cc_binary(@upb_bootstrap_compiler_test_repo//:a_stage1)
add_executable(CMakeProject_a_stage1 "")
add_executable(CMakeProject::a_stage1 ALIAS CMakeProject_a_stage1)
target_link_libraries(CMakeProject_a_stage1 PUBLIC
        "CMakeProject::b_stage1"
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_a_stage1 PUBLIC cxx_std_17)
target_sources(CMakeProject_a_stage1 PRIVATE
        "${TEST_DIRECTORY}/a.cc")

# proto_library(@upb_bootstrap_compiler_test_repo//:d_proto)
add_library(CMakeProject_d_proto INTERFACE)
target_sources(CMakeProject_d_proto INTERFACE
        "${TEST_DIRECTORY}/d.proto")
target_include_directories(CMakeProject_d_proto INTERFACE
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::d_proto ALIAS CMakeProject_d_proto)
