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
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "org_blosc_cblosc",
        strip_prefix = "c-blosc-1.21.6",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/Blosc/c-blosc/archive/v1.21.6.zip",
        ],
        sha256 = "1919c97d55023c04aa8771ea8235b63e9da3c22e3d2a68340b33710d19c2a2eb",
        build_file = Label("//third_party:org_blosc_cblosc/cblosc.BUILD.bazel"),
        system_build_file = Label("//third_party:org_blosc_cblosc/system.BUILD.bazel"),
        cmake_name = "Blosc",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:blosc": "Blosc::blosc",
        },
    )
