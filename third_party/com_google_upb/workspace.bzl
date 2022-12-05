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

# Must match grpc upb, see grpc/bazel/grpc_deps.bzl

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_upb",
        sha256 = "ac79e4540f04a6de945cd827b2e1be71f7d232c46f7d09621de2da1003c763d9",
        strip_prefix = "upb-f3a0cc49da29dbdbd09b3325c2834139540f00fa",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/protocolbuffers/upb/archive/f3a0cc49da29dbdbd09b3325c2834139540f00fa.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/f3a0cc49da29dbdbd09b3325c2834139540f00fa.tar.gz",  # main(2022-11-18)
        ],
        repo_mapping = {
            "@com_google_googleapis": "",  # Exclude googleapis; within upb it is only needed for test cases.
        },
    )
