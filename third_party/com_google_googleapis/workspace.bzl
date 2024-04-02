# Copyright 2023 The TensorStore Authors
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
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_googleapis",
        strip_prefix = "googleapis-aa829ea9860cdc474fccdf8f181f13d2a0eb2718",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/googleapis/googleapis/archive/aa829ea9860cdc474fccdf8f181f13d2a0eb2718.tar.gz",  # master(2023-10-04)
        ],
        sha256 = "93d98f5dcd22289e2bf0126b2f598b940c27a2e7ca74038930b0e273fba0632c",
        repo_mapping = {
            "@com_google_googleapis_imports": "@local_proto_mirror",
            "@com_google_protobuf_upb": "@com_google_protobuf",
        },
        cmake_name = "Googleapis",
        bazel_to_cmake = {
            "args": [
                "--exclude-target=//:build_gen",
                "--target=//google/api:all",
                "--target=//google/api/expr/v1alpha1:all",
                "--target=//google/rpc:all",
                "--target=//google/storage/v2:all",
            ],
        },
    )
