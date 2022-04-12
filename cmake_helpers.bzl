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
# When fallback is true, tests whether the package is found and, if not,
# uses FetchContent to download the required package.
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
        make_available = True,
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

# Adds raw text to the CMake file.
# Example:
#   cmake_raw(text="# Comment \n")
def cmake_raw(text = None):
    return

# Sets the section for the current file.  (Only useful for workspace.bzl files)
# The section controls the output order of subsequent CMake script.
# By convention, each file has a default section.
#
# Example:
#   cmake_set_section(section=20000)
#   cmake_raw(text="# B \n")
#   cmake_set_section(section=1)
#   cmake_raw(text="# A \n")

def cmake_set_section(section = None):
    return

def cmake_get_section():
    return 0
