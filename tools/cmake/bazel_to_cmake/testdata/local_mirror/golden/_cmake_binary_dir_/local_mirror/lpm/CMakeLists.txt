set(CMAKE_MESSAGE_INDENT "[lpm] ")

project("lpm")
execute_process(
  COMMAND ${Python3_EXECUTABLE} "${TEST_DIRECTORY}/bazel_to_cmake.py"
        --load-workspace "_workspace.pickle"
        --cmake-project-name lpm
        --cmake-binary-dir "${CMAKE_CURRENT_BINARY_DIR}"
        --bazel-repo-name local_proto_mirror
        --build-rules-output "${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND_ERROR_IS_FATAL ANY)
include("${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake")
