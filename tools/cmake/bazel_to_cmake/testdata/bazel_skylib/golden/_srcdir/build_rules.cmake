
# bazel_to_cmake wrote ${TEST_BINDIR}/config.h

# expand_template(@bazel_skylib_test_repo//:config2_h)
add_custom_command(
OUTPUT "${TEST_BINDIR}/config2.h"
COMMAND ${Python3_EXECUTABLE} "${SCRIPT_DIRECTORY}/bzl_library/expand_template.py"
        "${TEST_BINDIR}/config.h"
        "${TEST_BINDIR}/CMakeProject_config2_h.subs.json"
        "${TEST_BINDIR}/config2.h"
VERBATIM
DEPENDS "${SCRIPT_DIRECTORY}/bzl_library/expand_template.py" "${TEST_BINDIR}/CMakeProject_config2_h.subs.json" "${TEST_BINDIR}/config.h"
COMMENT "Generating ${TEST_BINDIR}/config2.h"
)
set_source_files_properties("${TEST_BINDIR}/config2.h" PROPERTIES GENERATED TRUE)
add_custom_target(CMakeProject_config2_h DEPENDS "${TEST_BINDIR}/config2.h")
target_include_directories(CMakeProject_config2_h INTERFACE "${PROJECT_BINARY_DIR}")

# genrule(@bazel_skylib_test_repo//:config_copy_rule)
add_custom_command(
OUTPUT
    "${TEST_BINDIR}/config3.h"
COMMAND ${CMAKE_COMMAND} -E copy "${TEST_BINDIR}/config2.h" "${TEST_BINDIR}/config3.h"
VERBATIM
DEPENDS
    "${TEST_BINDIR}/config2.h"
    "CMakeProject_config2_h"
WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_source_files_properties(
    "${TEST_BINDIR}/config3.h"
PROPERTIES GENERATED TRUE)
add_custom_target(genrule__CMakeProject_config_copy_rule DEPENDS
    "${TEST_BINDIR}/config3.h")
add_library(CMakeProject_config_copy_rule INTERFACE)
target_sources(CMakeProject_config_copy_rule INTERFACE
    "${TEST_BINDIR}/config3.h")
target_include_directories(CMakeProject_config_copy_rule INTERFACE
    "${PROJECT_BINARY_DIR}")
add_dependencies(CMakeProject_config_copy_rule genrule__CMakeProject_config_copy_rule)

# cc_library(@bazel_skylib_test_repo//:config_cc_library)
add_library(CMakeProject_config_cc_library INTERFACE)
target_link_libraries(CMakeProject_config_cc_library INTERFACE
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_config_cc_library INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_config_cc_library INTERFACE cxx_std_17)
add_dependencies(CMakeProject_config_cc_library "CMakeProject_config2_h" "CMakeProject_config_copy_rule")
add_library(CMakeProject::config_cc_library ALIAS CMakeProject_config_cc_library)
