
# bazel_to_cmake wrote ${TEST_BINDIR}/config.h

# expand_template(@bazel_skylib_test_repo//:config2_h)
add_custom_command(
OUTPUT "${TEST_BINDIR}/config2.h"
COMMAND ${Python3_EXECUTABLE} "${SCRIPT_DIRECTORY}/bzl_library/expand_template.py"
        "${TEST_BINDIR}/config.h"
        "${TEST_BINDIR}/CMakeProject_config2_h.subs.json"
        "${TEST_BINDIR}/config2.h"
DEPENDS "${TEST_BINDIR}/CMakeProject_config2_h.subs.json" "${TEST_BINDIR}/config.h" "${SCRIPT_DIRECTORY}/bzl_library/expand_template.py"
VERBATIM
)
add_custom_target(CMakeProject_config2_h DEPENDS "${TEST_BINDIR}/config2.h")

# genrule(@bazel_skylib_test_repo//:config_copy_rule)
add_custom_command(
  OUTPUT
    "${TEST_BINDIR}/config3.h"
  DEPENDS
    "${TEST_BINDIR}/config2.h"
    "CMakeProject_config2_h"
  COMMAND ${CMAKE_COMMAND} -E copy "${TEST_BINDIR}/config2.h" "${TEST_BINDIR}/config3.h"
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(genrule__CMakeProject_config_copy_rule DEPENDS
    "${TEST_BINDIR}/config3.h")
add_library(CMakeProject_config_copy_rule INTERFACE)
target_sources(CMakeProject_config_copy_rule INTERFACE
    "${TEST_BINDIR}/config3.h")
target_include_directories(CMakeProject_config_copy_rule INTERFACE
    "${PROJECT_BINARY_DIR}")
add_dependencies(CMakeProject_config_copy_rule genrule__CMakeProject_config_copy_rule)
