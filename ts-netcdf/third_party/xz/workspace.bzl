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
        name = "xz",
        strip_prefix = "xz-5.8.1",
        doc_homepage = "https://tukaani.org/xz/",
        doc_version = "5.8.1",
        urls = mirror_url("https://github.com/tukaani-project/xz/releases/download/v5.8.1/xz-5.8.1.tar.gz"),
        sha256 = "507825b599356c10dca1cd720c9d0d0c9d5400b9de300af00e4d1ea150795543",
        patches = [
            Label("//third_party:xz/patches/remove_have_config.diff"),
        ],
        patch_args = ["-p1"],
        build_file = Label("//third_party:xz/xz.BUILD.bazel"),
        system_build_file = Label("//third_party:xz/system.BUILD.bazel"),
        cmake_name = "LibLZMA",
        cmake_target_mapping = {
            "//:lzma": "LibLZMA::LibLZMA",
        },
        bazel_to_cmake = {},
    )
