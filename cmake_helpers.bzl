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

# These rules are intentionally empty; they exist to provide a framework
# for mapping between bazel deps and CMake rule deps. See bazel_to_cmake.py
# for actual implementation.
#
# Use:
#   load("//:cmake_helpers.bzl", "cmake_add_dep_mapping")
#   cmake_add_dep_mapping(bazel_target="@foo//:foo", cmake_target="foo")
#

# Uses CMake find_package(name) script to locate the package.
# When fallback is true, conditionally downloads the package.
def cmake_find_package(
        name = None,
        version = None,
        fallback = False,
        settings = None,
        **kwargs):
    return

# Uses CMake FetchContent script to locate the package.
def cmake_fetch_content_package(
        name = None,
        settings = None,
        **kwargs):
    return

# Uses CMake ExternalProject script to locate the package.
def cmake_external_project(
        name = None,
        settings = None,
        **kwargs):
    return

# Sets up a mapping from a bazel target to a cmake library.
# Example:
#   cmake_add_dep_mapping(target_mapping = {
#     "@com_github_nlohmann_json//:nlohmann_json": "nlohmann_json::nlohmann_json",
#   })
#
def cmake_add_dep_mapping(target_mapping = None):
    return

# Sets up a mapping from a bazel package to a cmake library.
# Generally assumes that the target mapping follows the pattern used by abseil CMake.
#
# cmake_use_absl_style_mapping(prefix_mapping={
#        "@foo_bar_bazel": "foo"
#     })
def cmake_use_absl_style_mapping(prefix_mapping = None):
    return

def cmake_raw(text):
    return
