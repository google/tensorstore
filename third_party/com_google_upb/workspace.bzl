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
        sha256 = "d0fe259d650bf9547e75896a1307bfc7034195e4ae89f5139814d295991ba681",
        strip_prefix = "upb-bef53686ec702607971bd3ea4d4fefd80c6cc6e8",
        urls = [
            "https://storage.googleapis.com/grpc-bazel-mirror/github.com/protocolbuffers/upb/archive/bef53686ec702607971bd3ea4d4fefd80c6cc6e8.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/bef53686ec702607971bd3ea4d4fefd80c6cc6e8.tar.gz",
        ],
    )
