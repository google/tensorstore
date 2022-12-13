
# bazel_to_cmake wrote _cmake_binary_dir_/config.h

add_custom_command(
OUTPUT "_cmake_binary_dir_/config2.h"
COMMAND ${Python3_EXECUTABLE} "${SCRIPT_DIRECTORY}/bzl_library/expand_template.py"
        "_cmake_binary_dir_/config.h"
        "_cmake_binary_dir_/CMakeProject_config2_h.subs.json"
        "_cmake_binary_dir_/config2.h"
DEPENDS "_cmake_binary_dir_/config.h" "${SCRIPT_DIRECTORY}/bzl_library/expand_template.py" "_cmake_binary_dir_/CMakeProject_config2_h.subs.json"
VERBATIM
)
add_custom_target(CMakeProject_config2_h DEPENDS "_cmake_binary_dir_/config2.h")

# @bazel_test_repo//:config_copy_rule
add_custom_command(
  OUTPUT "_cmake_binary_dir_/config3.h"
  DEPENDS "CMakeProject_config2_h" "_cmake_binary_dir_/config2.h"
  COMMAND ${CMAKE_COMMAND} -E copy "_cmake_binary_dir_/config2.h" "_cmake_binary_dir_/config3.h"
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(CMakeProject_config_copy_rule DEPENDS "_cmake_binary_dir_/config3.h")
