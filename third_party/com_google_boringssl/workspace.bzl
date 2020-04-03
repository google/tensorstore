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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_boringssl",
        urls = [
            # When updating, always use commit from master-with-bazel branch.
            "https://github.com/google/boringssl/archive/34693f02f6cf9ac7982778b761c16a27f32433c1.tar.gz",  # 2019-09-25
        ],
        sha256 = "633e2e806d01a07a20725d1e68fff0be96db18344ed4389c00de042dcd874cac",
        strip_prefix = "boringssl-34693f02f6cf9ac7982778b761c16a27f32433c1",
        system_build_file = Label("//third_party/com_google_boringssl:system.BUILD.bazel"),
    )
