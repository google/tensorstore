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

include(CMakeParseArguments)

# tensorstore_cc_library()
#
# CMake function to imitate Bazel's cc_library rule.
#
# Parameters:
# NAME: name of target (see Note)
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# PUBLIC: Add this so that this library will be exported under tensorstore::
# Also in IDE, target will appear in Tensorstore folder while non PUBLIC will be
#   in Tensorstore/internal.
# TESTONLY: When added, this target will only be built if BUILD_TESTING=ON.
#
# NOTES:
# * Modelled on the absl code.
# * By default, tensorstore_cc_library will always create a library named
#   tensorstore_${NAME}, and alias target tensorstore::${NAME}.
#   The tensorstore:: form should always be used.
#   This is to reduce namespace pollution.
#
# * Uses additional variables: TENSORSTORE_COMMON_INCLUDE_DIRS,
#   TENSORSTORE_DEFAULT_LINKOPTS, TENSORSTORE_ENABLE_INSTALL
#
#
# tensorstore_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
#
# tensorstore_cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     tensorstore::awesome   # not "awesome" !
#   PUBLIC
# )
#
# tensorstore_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     tensorstore::fantastic_lib
# )
#
# TODO: Implement "ALWAYSLINK"
function(tensorstore_cc_library)
  cmake_parse_arguments(TENSORSTORE_CC_LIB
    "DISABLE_INSTALL;PUBLIC;TESTONLY"
    "NAME"
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
    ${ARGN}
  )

  if(NOT TENSORSTORE_CC_LIB_PUBLIC AND TENSORSTORE_CC_LIB_TESTONLY AND
      NOT BUILD_TESTING)
    return()
  endif()

  if(TENSORSTORE_ENABLE_INSTALL)
    set(_NAME "${TENSORSTORE_CC_LIB_NAME}")
  else()
    set(_NAME "tensorstore_${TENSORSTORE_CC_LIB_NAME}")
  endif()

  # Check if this is a header-only library
  # Note that as of February 2019, many popular OS's (for example, Ubuntu
  # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
  # use list(FILTER...)
  set(TENSORSTORE_CC_SRCS "${TENSORSTORE_CC_LIB_SRCS}")
  foreach(src_file IN LISTS TENSORSTORE_CC_SRCS)
    if(${src_file} MATCHES ".*\\.(h|inc)")
      list(REMOVE_ITEM TENSORSTORE_CC_SRCS "${src_file}")
    endif()
  endforeach()

  if(TENSORSTORE_CC_SRCS STREQUAL "")
    set(TENSORSTORE_CC_LIB_IS_INTERFACE 1)
  else()
    set(TENSORSTORE_CC_LIB_IS_INTERFACE 0)
  endif()

  # NOTE: See this youtube talk on cmake targets: https://youtu.be/y7ndUhdQuU8

  # Determine this build target's relationship to the DLL. It's one of two things:
  # 1. "shared"  -- This is a shared library, perhaps on a non-windows platform
  #                 where DLL doesn't make sense.
  # 2. "static"  -- This target does not depend on the DLL and should be built
  #                 statically.
  if(BUILD_SHARED_LIBS)
    set(_build_type "shared")
  else()
    set(_build_type "static")
  endif()

  if(NOT TENSORSTORE_CC_LIB_IS_INTERFACE)
    add_library(${_NAME})
    target_sources(${_NAME} PRIVATE ${TENSORSTORE_CC_LIB_SRCS} ${TENSORSTORE_CC_LIB_HDRS})
    target_link_libraries(${_NAME}
    PUBLIC ${TENSORSTORE_CC_LIB_DEPS}
    PRIVATE
      ${TENSORSTORE_CC_LIB_LINKOPTS}
      ${TENSORSTORE_DEFAULT_LINKOPTS}
    )

    # Linker language "CXX" is always the correct linker language for static or
    # for shared libraries, we set it unconditionally.
    set_property(TARGET ${_NAME} PROPERTY LINKER_LANGUAGE "CXX")

    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${TENSORSTORE_COMMON_INCLUDE_DIRS}>"
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_options(${_NAME}
      PRIVATE ${TENSORSTORE_CC_LIB_COPTS})

    target_compile_definitions(${_NAME} PUBLIC ${TENSORSTORE_CC_LIB_DEFINES})

    # Tensorstore libraries require C++17 as the current minimum standard.
    # Top-level application CMake projects should ensure a consistent C++
    # standard for all compiled sources by setting CMAKE_CXX_STANDARD.
    target_compile_features(${_NAME} PUBLIC cxx_std_17)

  else()
    # Generating header-only library
    add_library(${_NAME} INTERFACE)
    target_include_directories(${_NAME}
      INTERFACE
        "$<BUILD_INTERFACE:${TENSORSTORE_COMMON_INCLUDE_DIRS}>"
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      )

    target_link_libraries(${_NAME}
      INTERFACE
        ${TENSORSTORE_CC_LIB_DEPS}
        ${TENSORSTORE_CC_LIB_LINKOPTS}
        ${TENSORSTORE_DEFAULT_LINKOPTS}
    )
    target_compile_definitions(${_NAME} INTERFACE ${TENSORSTORE_CC_LIB_DEFINES})

    # Tensorstore libraries require C++17 as the current minimum standard.
    # Top-level application CMake projects should ensure a consistent C++
    # standard for all compiled sources by setting CMAKE_CXX_STANDARD.
    target_compile_features(${_NAME} INTERFACE cxx_std_17)

  endif()

  # Add tensorstore:: alias.
  add_library(tensorstore::${TENSORSTORE_CC_LIB_NAME} ALIAS ${_NAME})

  # Add install target
  if(NOT TENSORSTORE_CC_LIB_TESTONLY AND TENSORSTORE_ENABLE_INSTALL)
    install(TARGETS ${_NAME} EXPORT ${PROJECT_NAME}Targets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
  endif()

  # Generate pkg-config for libraries.
  if(NOT TENSORSTORE_CC_LIB_TESTONLY AND TENSORSTORE_ENABLE_INSTALL)
    set(PC_VERSION "head")
    foreach(dep ${TENSORSTORE_CC_LIB_DEPS})
      if(${dep} MATCHES "^tensorstore::(.*)")
      # Join deps with commas.
        if(PC_DEPS)
          set(PC_DEPS "${PC_DEPS},")
        endif()
        set(PC_DEPS "${PC_DEPS} tensorstore_${CMAKE_MATCH_1} = ${PC_VERSION}")
      endif()
    endforeach()
    foreach(cflag ${TENSORSTORE_CC_LIB_COPTS})
      if(${cflag} MATCHES "^(-Wno|/wd)")
        # These flags are needed to suppress warnings that might fire in our headers.
        set(PC_CFLAGS "${PC_CFLAGS} ${cflag}")
      elseif(${cflag} MATCHES "^(-W|/w[1234eo])")
        # Don't impose our warnings on others.
      else()
        set(PC_CFLAGS "${PC_CFLAGS} ${cflag}")
      endif()
    endforeach()

    file(GENERATE
         OUTPUT "${CMAKE_BINARY_DIR}/lib/pkgconfig/tensorstore_${_NAME}.pc"
         CONTENT "\
prefix=${CMAKE_INSTALL_PREFIX}\n\
exec_prefix=\${prefix}\n\
libdir=${CMAKE_INSTALL_FULL_LIBDIR}\n\
includedir=${CMAKE_INSTALL_FULL_INCLUDEDIR}\n\
\n\
Name: tensorstore_${_NAME}\n\
Description: Tensorstore ${_NAME} library\n\
URL: https://google.github.io/tensorstore/\n\
Version: ${PC_VERSION}\n\
Requires:${PC_DEPS}\n\
Libs: -L\${libdir} $<JOIN:${TENSORSTORE_CC_LIB_LINKOPTS}, > $<$<NOT:$<BOOL:${TENSORSTORE_CC_LIB_IS_INTERFACE}>>:-labsl_${_NAME}>\n\
Cflags: -I\${includedir}${PC_CFLAGS}\n")

    install(FILES "${CMAKE_BINARY_DIR}/lib/pkgconfig/tensorstore_${_NAME}.pc"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
  endif()

endfunction()


# tensorstore_cc_test()
#
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# NOTES:
# * Modelled on the absl code.
# * By default, tensorstore_cc_test will always create a binary named
#   tensorstore_${NAME}.
#   This will also add it to ctest list as tensorstore_${NAME}.
#
# * Uses additional variables: TENSORSTORE_COMMON_INCLUDE_DIRS
#   GMOCK_INCLUDE_DIRS, GTEST_INCLUDE_DIRS
#
# Usage:
#
# tensorstore_cc_test(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     tensorstore::awesome
#     GTest::gmock
#     GTest::gtest_main
# )
function(tensorstore_cc_test)
  if(NOT BUILD_TESTING )
    return()
  endif()

  cmake_parse_arguments(TENSORSTORE_CC_TEST
    ""
    "NAME"
    "SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
    ${ARGN}
  )

  set(_NAME "tensorstore_${TENSORSTORE_CC_TEST_NAME}")

  add_executable(${_NAME} "")
  target_sources(${_NAME} PRIVATE ${TENSORSTORE_CC_TEST_SRCS})
  target_include_directories(${_NAME}
    PUBLIC ${TENSORSTORE_COMMON_INCLUDE_DIRS}
    PRIVATE ${GMOCK_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS}
  )

  target_compile_definitions(${_NAME}
    PUBLIC
      ${TENSORSTORE_CC_TEST_DEFINES}
  )

  target_compile_options(${_NAME}
    PRIVATE ${TENSORSTORE_CC_TEST_COPTS}
  )

  target_link_libraries(${_NAME}
    PUBLIC ${TENSORSTORE_CC_TEST_DEPS}
    PRIVATE ${TENSORSTORE_CC_TEST_LINKOPTS}
  )

  # Tensorstore libraries require C++17 as the current minimum standard.
  # Top-level application CMake projects should ensure a consistent C++
  # standard for all compiled sources by setting CMAKE_CXX_STANDARD.
  target_compile_features(${_NAME} PUBLIC cxx_std_17)

  add_test(NAME ${_NAME} COMMAND ${_NAME})
endfunction()


# tensorstore_proto_cc_library()
#
# CMake function to imitate Bazel's proto_cc_library rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# PROTOS: List of proto source files for the library
# COPTS: List of private compile options
# DEPS: List of other libraries to be linked in to the binary targets
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# Usage:
#
# tensorstore_proto_cc_library(
#   NAME
#     cool_cc_proto
#   PROTOS
#     "cool.proto"
# )
#
function(tensorstore_proto_cc_library)

  cmake_parse_arguments(TENSORSTORE_PROTO_CC_LIBRARY
    ""
    "NAME"
    "PROTOS;DEPS;COPTS;DEFINES;LINKOPTS"
    ${ARGN}
  )

  ts_protobuf_generate_cpp(_proto_srcs _proto_hdrs
      ${TENSORSTORE_PROTO_CC_LIBRARY_PROTOS})

  set_source_files_properties(${_proto_srcs} ${_proto_hdrs} PROPERTIES GENERATED TRUE)

  tensorstore_cc_library(
    NAME ${TENSORSTORE_PROTO_CC_LIBRARY_NAME}
    SRCS "${_proto_srcs}"
    HDRS "${_proto_hdrs}"
    DEPS "${TENSORSTORE_PROTO_CC_LIBRARY_DEPS}"
    COPTS "${TENSORSTORE_PROTO_CC_LIBRARY_COPTS}"
    DEFINES "${TENSORSTORE_PROTO_CC_LIBRARY_DEFINES}"
    LINKOPTS "${TENSORSTORE_PROTO_CC_LIBRARY_LINKOPTS}"
  )

endfunction()


# check_target(target)
#   Errors if the targetg does not exist.
function(check_target my_target)
  if(NOT TARGET ${my_target})
    message(FATAL_ERROR " TENSORSTORE: compiling tensorstore requires a
                   ${my_target} CMake target in your project,
                   see CMake/README.md for more details")
  endif(NOT TARGET ${my_target})
endfunction()


# maybe_add_alias(target alias)
#   Attempts to add an alias for a target if the target does not exist.
function(maybe_add_alias my_target my_alias)
  if(NOT TARGET ${my_alias})
    if(TARGET ${my_target})
      add_library(${my_target} ALIAS ${my_alias})
    endif(TARGET ${my_target})
  endif(NOT TARGET ${my_alias})
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


# Helpers for debugging CMake

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
