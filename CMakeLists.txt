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

# NOTE: This is still very much a work in progress; It is not yet expected to
# build.

cmake_minimum_required(VERSION 3.24)
project(tensorstore LANGUAGES CXX)

# Enable CMP0077 (option honors variables) for subprojects.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Enable CMP0126 for subprojects to prevent `set(<var> CACHE)`
# overriding a prior `set(<var>)` call.
set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)

# Do not add the current directory to build commands by default.
# This is defensive, as the setting may be inherited from parents,
# and it has been known to cause errors in Abseil CMake builds.
set(CMAKE_INCLUDE_CURRENT_DIR OFF)

# Set all outputs to a single /bin directory.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)

if(PROJECT_IS_TOP_LEVEL)
  include(CTest)
  enable_testing()
else()
  # Exclude targets from ALL when built as a sub-project.
  set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/tools/cmake)

##
## Using tensorstore targets
##
## all public tensorstore targets are
## exported with the tensorstore:: prefix
##
## DO NOT rely on the internal targets outside of the prefix

## Abseil requires PIC code; we generally use the same.
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

find_package(Threads REQUIRED)

if(MSVC AND CMAKE_CXX_COMPILER_LAUNCHER MATCHES "sccache")
  # sccache is incompatible with the /Zi option that CMake enables by default
  #
  # https://github.com/mozilla/sccache#usage
  #
  # Note that this unavoidably affects the entire build, not just this
  # sub-project.
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  endif()
endif()

include(bazel_to_cmake)
include(btc_protobuf)

bazel_to_cmake(
  --cmake-project-name tensorstore
  --bazel-repo-name com_google_tensorstore
  "--include-package="
  "--include-package=tensorstore/**"
  "--exclude-package=tensorstore/examples/python/**"
  --ignore-library "//third_party:python/python_configure.bzl"
  --ignore-library "//docs:doctest.bzl"
  --ignore-library "//bazel:non_compile.bzl"
  --ignore-library "@com_google_protobuf//:protobuf.bzl"
  --define "grpc_no_xds=true"
  --bazelrc .bazelrc
  --module bazel_to_cmake.bzl_library.grpc_generate_cc
  --module bazel_to_cmake.bzl_library.local_mirror
  --module bazel_to_cmake.bzl_library.rules_nasm
  --module bazel_to_cmake.bzl_library.third_party_http_archive
  --module bazel_to_cmake.bzl_library.upb_proto_library
)

if(TENSORSTORE_DEBUG_DUMP_TARGETS)
  include(TensorstoreDebugHelpers)
  dump_cmake_targets(${CMAKE_CURRENT_SOURCE_DIR})
endif()
