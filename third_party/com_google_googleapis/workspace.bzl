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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_googleapis",
        strip_prefix = "googleapis-32bc03653260356351854429bd7e2dfbf670d352",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/googleapis/googleapis/archive/32bc03653260356351854429bd7e2dfbf670d352.tar.gz",  # master(2024-09-10)
        ],
        sha256 = "46ca6d9a6349c3845334dde2d55d482a11e7c1072a9085b89b6c1e94cdeb2d3e",
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
