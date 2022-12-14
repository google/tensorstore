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

# source = https://aomedia.googlesource.com/aom/

def repo():
    maybe(
        third_party_http_archive,
        name = "org_aomedia_aom",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/aomedia.googlesource.com/aom/+archive/d730cef03ac754f2b6a233e926cd925d8ce8de81.tar.gz",  # main(2022-11-01)
            # "https://aomedia.googlesource.com/aom/+archive/d730cef03ac754f2b6a233e926cd925d8ce8de81.tar.gz
        ],
        # googlesource does not cache archive files; the sha256 is only valid for the mirror.
        sha256 = "7f87960be61edb846e25b7d10e4e41aef6d652f62270e19172e7cafae9b536ad",
        build_file = Label("//third_party:org_aomedia_aom/libaom.BUILD.bazel"),
        cmake_name = "aom",
        cmake_languages = ["ASM"],
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:libaom": "AOM::AOM",
        },
        cmake_package_redirect_libraries = {
            "AOM": "AOM::AOM",
        },
    )
