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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

#   Canonical location for dav1d codec is https://code.videolan.org/videolan/dav1d

def repo():
    maybe(
        third_party_http_archive,
        name = "org_videolan_dav1d",
        sha256 = "fa635e2bdb25147b1384007c83e15de44c589582bb3b9a53fc1579cb9d74b695",
        strip_prefix = "dav1d-1.5.1",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/videolan/dav1d/archive/1.5.1.tar.gz",
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
