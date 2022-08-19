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


# Helpers for debugging CMake

# check_target(target)
#   Errors if the targetg does not exist.
function(check_target my_target)
  if(NOT TARGET ${my_target})
    message(FATAL_ERROR " TENSORSTORE: compiling tensorstore requires a
                   ${my_target} CMake target in your project,
                   see CMake/README.md for more details")
  endif(NOT TARGET ${my_target})
endfunction()


# check_absl_target(target)
#   Attempts to add an alias for an absl namespace target
#   before running check_target(target)
function(check_absl_target my_target)
  string(FIND ${my_target} "::" _has_namespace)
  if(${_has_namespace})
    string(REPLACE "::" "_" _my_alias ${my_target})
    maybe_add_alias(${my_target} ${_my_alias})
  endif(${_has_namespace})

  check_target("${my_target}")
endfunction()



# dump_cmake_variables()
#   Dumps all the CMAKE variables.
function(dump_cmake_variables)
  # https://stackoverflow.com/questions/9298278/cmake-print-out-all-accessible-variables-in-a-script
  get_cmake_property(_variableNames VARIABLES)
  list (SORT _variableNames)
  foreach (_variableName ${_variableNames})
    if (ARGV0)
      unset(MATCHED)
      string(REGEX MATCH ${ARGV0} MATCHED ${_variableName})
      if (NOT MATCHED)
         continue()
      endif()
    endif()
    message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endfunction()

# dump_cmake_targets( <DIRECTORY> )
#   Dumps all the CMAKE targets under the <DIRECTORY>.
function(dump_cmake_targets directory)
  get_property(imported_targets DIRECTORY ${directory} PROPERTY IMPORTED_TARGETS)
  foreach(_target ${imported_targets})
    message(STATUS "+ ${_target}")
  endforeach()

  get_property(dir_targets DIRECTORY ${directory} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_target ${dir_targets})
    get_target_property(_type ${_target} TYPE)
    message(STATUS "+ ${_target}  ${_type}")
  endforeach()

  get_property(sub_directories DIRECTORY ${directory} PROPERTY SUBDIRECTORIES)
  foreach(directory ${sub_directories})
    dump_cmake_targets(${directory})
  endforeach()
endfunction()
