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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_storagetestbench",
        strip_prefix = "storage-testbench-f197d6560cd325880888017c1c8d15cb923460bd",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/googleapis/storage-testbench/archive/f197d6560cd325880888017c1c8d15cb923460bd.tar.gz",  # main(2023-11-08)
        ],
        sha256 = "52be6755152d228a5a0a6dbabc4784ef7ad299dec3be7631ce6fe2f73bf191bb",
        build_file = Label("//third_party:com_google_storagetestbench/storagetestbench.BUILD.bazel"),
    )
