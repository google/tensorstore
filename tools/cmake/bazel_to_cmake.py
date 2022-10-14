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
"""Generates CMake build rules from a Bazel workspace.

The top-level `CMakeLists.txt` invokes this program at configure time using
`execute_process`.  It has access to all of the relevant CMake options, and
generates CMake rules specific to the current configuration.  It also outputs
appropriate dependency information to ensure that all of the relevant Bazel
build configuration files are listed as dependencies.

The WORKSPACE and BUILD files (which use the Starlark language) are evaluated as
Python, since Starlark is (almost) a subset of Python.

Substitute definitions of native Bazel rules, and neceessary third-party
Starlark packages like `@bazel_skylib`, are provided that generate CMake build
rules instead.

When evaluating the WORKSPACE, calls to `third_party_http_archive` are converted
to `FetchContent` commands.

The current CMake configuration is used to synthesize an appropriate set of
`@platforms` constraints used to evaluate `select` expressions.
"""

import sys

import bazel_to_cmake.main

if __name__ == "__main__":
  sys.exit(bazel_to_cmake.main.main())
