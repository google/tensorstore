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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

# Use libpng from a github mirror rather than https://git.code.sf.net/p/libpng/code
# since it's much easier to download.
def repo():
    maybe(
        third_party_http_archive,
        # Note: The generic name "png" is used in place of the more canonical
        # "org_libjpng" because this repository may actually refer to the
        # system png.
        name = "png",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/glennrp/libpng/archive/v1.6.42.tar.gz",
        ],
        sha256 = "fe89de292e223829859d21990d9c4d6b7e30e295a268f6a53a022611aa98bd67",
        strip_prefix = "libpng-1.6.42",
        build_file = Label("//third_party:png/png.BUILD.bazel"),
        system_build_file = Label("//third_party:png/system.BUILD.bazel"),
        cmake_name = "PNG",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@png//:png": "PNG::PNG",
        },
    )
