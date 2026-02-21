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
        strip_prefix = "googleapis-1937a1552cb031159749738f6d062f084ea94894",
        doc_version = "20260206-a65054d",
        urls = mirror_url("https://github.com/googleapis/googleapis/archive/1937a1552cb031159749738f6d062f084ea94894.tar.gz"),  # master(2026-02-18)
        sha256 = "164641dfc31c148742f055ab8b7a07efd39862b4cd2e6fa35b5f498a30ee07bb",
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
