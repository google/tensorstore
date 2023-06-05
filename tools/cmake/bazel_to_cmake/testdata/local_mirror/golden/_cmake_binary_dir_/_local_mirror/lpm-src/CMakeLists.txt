set(CMAKE_MESSAGE_INDENT "[lpm] ")

file(DOWNLOAD "https://raw.githubusercontent.com/bufbuild/protoc-gen-validate/2682ad06cca00550030e177834f58a2bc06eb61e/validate/validate.proto"
       "${CMAKE_CURRENT_SOURCE_DIR}/validate.proto"
     EXPECTED_HASH "SHA256=bf7ca2ac45a75b8b9ff12f38efd7f48ee460ede1a7919d60c93fad3a64fc2eee"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "${TEST_DIRECTORY}/a.cc"
       "${CMAKE_CURRENT_SOURCE_DIR}/a.cc"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

project("lpm")
execute_process(
  COMMAND ${Python3_EXECUTABLE} "run_bazel_to_cmake.py"
        --load-workspace "_workspace.pickle"
        --cmake-project-name lpm
        --cmake-binary-dir "${CMAKE_CURRENT_BINARY_DIR}"
        --bazel-repo-name local_proto_mirror
        --build-rules-output "${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND_ERROR_IS_FATAL ANY)
include("${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake")
