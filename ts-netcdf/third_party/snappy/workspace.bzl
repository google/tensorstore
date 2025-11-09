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
        urls = mirror_url("https://github.com/google/snappy/archive/6f99459b5b837fa18abb1be317d3ac868530f384.tar.gz"),  # main(2025-08-19)
        sha256 = "787aba190f05da2ac5e77e847896baf64b91905489c7ede1d09b01b363b2ff81",
        strip_prefix = "snappy-6f99459b5b837fa18abb1be317d3ac868530f384",
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
