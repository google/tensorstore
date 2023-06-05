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

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_storagetestbench",
        strip_prefix = "storage-testbench-738a36086109b051fd83059ad79d5a56a57af93a",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/googleapis/storage-testbench/archive/738a36086109b051fd83059ad79d5a56a57af93a.tar.gz",  # main(2023-05-22)
        ],
        sha256 = "078239895de8cba2864396bfd500b66cb0191be9751d20c37726f6f40c84fab1",
        build_file = Label("//third_party:com_google_storagetestbench/storagetestbench.BUILD.bazel"),
    )
