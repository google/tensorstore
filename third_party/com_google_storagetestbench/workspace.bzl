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
        strip_prefix = "storage-testbench-dffecd261771a124766e02f8a3410bdc5ace0401",
        urls = mirror_url("https://github.com/googleapis/storage-testbench/archive/dffecd261771a124766e02f8a3410bdc5ace0401.tar.gz"),  # main(2026-06-20)
        sha256 = "a50e7e3f25faef863bdc3d448ed2d1c31c0b29f5177360796c8b9a2aec2b04f1",
        build_file = Label("//third_party:com_google_storagetestbench/storagetestbench.BUILD.bazel"),
    )
