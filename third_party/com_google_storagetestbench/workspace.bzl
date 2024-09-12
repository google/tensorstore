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
load("//third_party:repo.bzl", "third_party_http_archive")

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_storagetestbench",
        strip_prefix = "storage-testbench-bc7738b9f0c737c198d67e03a12d21600c3e771b",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/googleapis/storage-testbench/archive/bc7738b9f0c737c198d67e03a12d21600c3e771b.tar.gz",  # main(2024-09-10)
        ],
        sha256 = "7a71839be3c5e0502cb699f4bf6f22f878cce7af4e7a593a6b2a0d1002bee873",
        build_file = Label("//third_party:com_google_storagetestbench/storagetestbench.BUILD.bazel"),
    )
