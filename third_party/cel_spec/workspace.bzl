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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

# Should be compatible with grpc/bazel/grpc_deps.bzl c-ares.

def repo():
    maybe(
        third_party_http_archive,
        name = "cel-spec",
        sha256 = "13234d9622ef4b7660b6259498a6f02f001fd260d42fa21e67b28fd4c4e0d344",
        strip_prefix = "cel-spec-0.16.2",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/google/cel-spec/archive/v0.16.2.zip",
        ],
        repo_mapping = {
            "@io_bazel_rules_go": "@local_proto_mirror",
        },
        cmake_name = "cel-spec",
        bazel_to_cmake = {
            "args": [
                "--target=//proto/cel/expr:expr_proto",
            ] + [
                "--target=//proto/cel/expr:" + p + "_proto"
                for p in _PROTO_TARGETS
            ] + [
                "--target=//proto/cel/expr:" + p + "_cc_proto"
                for p in _PROTO_TARGETS
            ] + [
                "--target=//proto/cel/expr:" + p + "_proto__upb_library"
                for p in _PROTO_TARGETS
            ] + [
                "--target=//proto/cel/expr:" + p + "_proto__upbdefs_library"
                for p in _PROTO_TARGETS
            ],
        },
    )

_PROTO_TARGETS = [
    "syntax",
    "checked",
    "value",
    "eval",
    "explain",
]
