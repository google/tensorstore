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
            "https://github.com/glennrp/libpng/archive/v1.6.37.tar.gz",
        ],
        sha256 = "ca74a0dace179a8422187671aee97dd3892b53e168627145271cad5b5ac81307",
        strip_prefix = "libpng-1.6.37",
        build_file = Label("//third_party:png/png.BUILD.bazel"),
        system_build_file = Label("//third_party:png/system.BUILD.bazel"),
        cmake_name = "PNG",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@png//:png": "PNG::PNG",
        },
    )
