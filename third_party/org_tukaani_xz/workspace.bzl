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

load("//third_party:repo.bzl", "third_party_http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "org_tukaani_xz",
        strip_prefix = "xz-5.3.3alpha",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/tukaani.org/xz/xz-5.3.3alpha.tar.gz",
            "https://tukaani.org/xz/xz-5.3.3alpha.tar.gz",
        ],
        sha256 = "c6d4789a79bab565440784db2e132d6bf03b2e75dd6d66a8102cf002d8dfe926",
        build_file = Label("//third_party:org_tukaani_xz/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:org_tukaani_xz/system.BUILD.bazel"),
        doc_homepage = "https://tukaani.org/xz/",
        cmake_name = "LibLZMA",
        cmake_target_mapping = {
            "@org_tukaani_xz//:liblzma": "LibLZMA::LibLZMA",
        },
        bazel_to_cmake = {},
    )
