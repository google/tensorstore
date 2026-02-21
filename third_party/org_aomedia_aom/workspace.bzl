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
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

# source = https://aomedia.googlesource.com/aom/
# See also https://chromium.googlesource.com/chromium/src/+/refs/heads/main/third_party/libaom
def repo():
    maybe(
        third_party_http_archive,
        name = "org_aomedia_aom",
        doc_version = "3.13.1",
        urls = mirror_url("https://aomedia.googlesource.com/aom/+archive/d772e334cc724105040382a977ebb10dfd393293.tar.gz"),
        # googlesource does not cache archive files; the sha256 is only valid for the mirror.
        sha256 = "5a36e9eaeaa4b9a437db6b1285dd12c038fa541ea716d9a68bf2a0de9d609eb6",
        build_file = Label("//third_party:org_aomedia_aom/libaom.BUILD.bazel"),
        cmake_name = "aom",
        cmake_languages = ["ASM"],
        bazel_to_cmake = {
            "aliased_targets_only": True,
        },
        cmake_target_mapping = {
            "//:libaom": "AOM::AOM",
        },
        cmake_package_redirect_libraries = {
            "AOM": "AOM::AOM",
        },
    )
