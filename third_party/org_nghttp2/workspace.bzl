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
        name = "org_nghttp2",
        strip_prefix = "nghttp2-1.66.0",
        urls = mirror_url("https://github.com/nghttp2/nghttp2/archive/v1.66.0.tar.gz"),
        sha256 = "e2505d08aafac303ed8e3b94bf8caafa55ddcc198f89b254bb79a61a379498ea",
        build_file = Label("//third_party:org_nghttp2/nghttp2.BUILD.bazel"),
        system_build_file = Label("//third_party:org_nghttp2/system.BUILD.bazel"),
        # https://github.com/nghttp2/nghttp2/blob/master/CMakeLists.txt
        cmake_name = "NGHTTP2",
        cmake_target_mapping = {
            "@org_nghttp2//:nghttp2": "NGHTTP2::NGHTTP2",
        },
        bazel_to_cmake = {},
        cmake_package_redirect_libraries = {
            "NGHTTP2": "NGHTTP2::NGHTTP2",
        },
    )
