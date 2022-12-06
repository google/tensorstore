# Copyright 2022 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Defines the `bazel_to_cmake` macro and related variables.

set(_bazel_to_cmake_dir "${CMAKE_CURRENT_LIST_DIR}")

macro(bazel_to_cmake)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  # Collect all variables as a JSON object.  JSON is used in order to
  # properly handle special characters.
  get_property(_bazel_to_cmake_vars DIRECTORY PROPERTY VARIABLES)
  # Add empty string as initial key/value pair to simplify commas.
  set(_bazel_to_cmake_vars_json "{\"\":\"\"")
  foreach(x ${_bazel_to_cmake_vars})
    if (NOT "${x}" MATCHES "^_")
      string(REPLACE "\\" "\\\\" value "${x}")
      string(REPLACE "\"" "\\\"" value "${value}")
      string(APPEND _bazel_to_cmake_vars_json ",\n\"${value}\":")
      string(REPLACE "\\" "\\\\" value "${${x}}")
      string(REPLACE "\"" "\\\"" value "${value}")
      string(REPLACE "\n" "\\n" value "${value}")
      string(REPLACE "\t" "\\t" value "${value}")
      string(REPLACE "\r" "\\r" value "${value}")
      string(APPEND _bazel_to_cmake_vars_json "\"${value}\"")
    endif()
  endforeach()
  string(APPEND _bazel_to_cmake_vars_json "}")

  # Write vars to a file rather than passing them on the command line
  # directly to avoid length limits.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/bazel_to_cmake_vars.json" "${_bazel_to_cmake_vars_json}")
  unset(_bazel_to_cmake_vars_json)
  unset(_bazel_to_cmake_vars)

  if (${BAZEL_TO_CMAKE_COMMAND})
    set(_bazel_to_cmake_command "${BAZEL_TO_CMAKE_COMMAND}")
  else()
    set(_bazel_to_cmake_command "${_bazel_to_cmake_dir}/bazel_to_cmake.py")
  endif()

  set(_bazel_to_cmake_args "${ARGN}")

  message(VERBOSE "Invoking bazel_to_cmake Using:")
  message(VERBOSE "WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}")
  message(VERBOSE "${Python3_EXECUTABLE} ${_bazel_to_cmake_command}")
  message(VERBOSE "--cmake-vars ${CMAKE_CURRENT_BINARY_DIR}/bazel_to_cmake_vars.json")
  message(VERBOSE "--cmake-binary-dir ${CMAKE_CURRENT_BINARY_DIR}")
  message(VERBOSE "--build-rules-output ${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake")
  message(VERBOSE "--save-workspace ${CMAKE_CURRENT_BINARY_DIR}/bazel_workspace_info.pickle")
  string(REPLACE ";" " " _bazel_to_cmake_args_str "${_bazel_to_cmake_args}")
  message(VERBOSE "${_bazel_to_cmake_args_str}")
  unset(_bazel_to_cmake_args_str)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${_bazel_to_cmake_command}"
      --cmake-vars "${CMAKE_CURRENT_BINARY_DIR}/bazel_to_cmake_vars.json"
      --cmake-binary-dir "${CMAKE_CURRENT_BINARY_DIR}"
      --build-rules-output "${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake"
      --save-workspace "${CMAKE_CURRENT_BINARY_DIR}/bazel_workspace_info.pickle"
      ${_bazel_to_cmake_args}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )

  unset(_bazel_to_cmake_command)
  unset(_bazel_to_cmake_args)

  include("${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake")
endmacro()

# Defines the `bazel_to_cmake_needed_library` link feature, which
# provides an equivalent of Bazel's `alwayslink = True` option for
# shared libraries.  CMake's builtin `WHOLE_ARCHIVE` link feature
# provides the corresponding behavior for static libraries.
set(CMAKE_LINK_LIBRARY_USING_bazel_to_cmake_needed_library_SUPPORTED TRUE)

if(CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF")
  set(CMAKE_LINK_LIBRARY_USING_bazel_to_cmake_needed_library
    "LINKER:--push-state,--no-as-needed"
    "<LINK_ITEM>"
    "LINKER:--pop-state")
elseif(CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
  set(CMAKE_LINK_LIBRARY_USING_bazel_to_cmake_needed_library
    "LINKER:-needed_library,<LINK_ITEM>")
else()
  # Assume that all libraries are linked by default.
  set(CMAKE_LINK_LIBRARY_USING_bazel_to_cmake_needed_library
    "<LINK_ITEM>")
endif()
