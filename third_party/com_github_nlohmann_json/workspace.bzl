# Copyright 2020 The TensorStore Authors
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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//:cmake_helpers.bzl", "cmake_add_dep_mapping", "cmake_fetch_content_package", "cmake_raw")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_nlohmann_json",
        urls = [
            "https://github.com/nlohmann/json/releases/download/v3.10.5/include.zip",
        ],
        sha256 = "b94997df68856753b72f0d7a3703b7d484d4745c567f3584ef97c96c25a5798e",
        build_file = Label("//third_party:com_github_nlohmann_json/bundled.BUILD.bazel"),
    )

cmake_add_dep_mapping(target_mapping = {
    "@com_github_nlohmann_json//:nlohmann_json": "nlohmann_json::nlohmann_json",
})

cmake_fetch_content_package(
    name = "com_github_nlohmann_json",
    configure_command = "",
    build_command = "",
    make_available = False,
)

cmake_raw(
    text = """
# nlohmann_json install doesn't work with FetchContent. :/

FetchContent_GetProperties(com_github_nlohmann_json)
if(NOT com_github_nlohmann_json_POPULATED)
  FetchContent_Populate(com_github_nlohmann_json)
endif()

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
      "${com_github_nlohmann_json_SOURCE_DIR}/single_include")

add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
""",
)
