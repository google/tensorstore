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
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "googleapis",
        strip_prefix = "googleapis-136201b66f70829232b7ec63fa6e35ca765bcacb",
        urls = mirror_url("https://github.com/googleapis/googleapis/archive/136201b66f70829232b7ec63fa6e35ca765bcacb.tar.gz"),  # master(2025-11-17)
        sha256 = "ead761dae7673cdfc46425d22e1d2becbd4519e2a75cb78d873ec67c53c75e8b",
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
                "--target=//google/iam/credentials/v1:all",
            ],
        },
    )
