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

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_storagetestbench",
        strip_prefix = "storage-testbench-b5c8fece39022c5c976a3db1fbdce0bf2c2c7cd9",
        urls = mirror_url("https://github.com/googleapis/storage-testbench/archive/b5c8fece39022c5c976a3db1fbdce0bf2c2c7cd9.tar.gz"),  # main(2026-06-12)
        sha256 = "ceab9780e99225648dc06df431ea69dc9fe12deb76a2ec465ebb9c63a7c67bcf",
        build_file = Label("//third_party:com_google_storagetestbench/storagetestbench.BUILD.bazel"),
    )
