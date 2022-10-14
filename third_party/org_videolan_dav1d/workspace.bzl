# Copyright 2021 The TensorStore Authors
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

#   Canonical location for dav1d codec is https://code.videolan.org/videolan/dav1d

def repo():
    maybe(
        third_party_http_archive,
        name = "org_videolan_dav1d",
        sha256 = "047b8229511a82b5718a1d34c86c067b078efd02f602986d2ed09b23182ec136",
        strip_prefix = "dav1d-1.0.0",
        urls = [
            "https://github.com/videolan/dav1d/archive/1.0.0.tar.gz",
        ],
        build_file = Label("//third_party:org_videolan_dav1d/dav1d.BUILD.bazel"),
        cmake_name = "dav1d",
        bazel_to_cmake = {},
        cmake_languages = ["ASM_NASM"],
        cmake_target_mapping = {
            "@org_videolan_dav1d//:dav1d": "DAV1D::DAV1D",
        },
        cmake_package_redirect_libraries = {
            "DAV1D": "DAV1D::DAV1D",
        },
    )
