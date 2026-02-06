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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "snappy",
        doc_version = "1.2.2-20250426-6af9287",
        urls = mirror_url("https://github.com/google/snappy/archive/da459b5263676ccf0dc65a3fcf93fb876e09baac.tar.gz"),  # main(2026-02-06)
        sha256 = "8fceeed9ad799d1e5a82fe89574e598db5a89b18ff828cd18011031f8a6d634e",
        strip_prefix = "snappy-da459b5263676ccf0dc65a3fcf93fb876e09baac",
        build_file = Label("//third_party:snappy/snappy.BUILD.bazel"),
        system_build_file = Label("//third_party:snappy/system.BUILD.bazel"),
        cmake_name = "Snappy",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:snappy": "Snappy::Snappy",
            "//:snappy-c": "Snappy::Snappy_c",
        },
        cmake_package_redirect_libraries = {
            "Snappy": "Snappy::Snappy_c",
        },
    )
