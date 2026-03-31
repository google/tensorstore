# Copyright 2026 The TensorStore Authors
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

"""
Local copies of envoyproxy/protoc-gen-validate files.

From: https://github.com/envoyproxy/protoc-gen-validate
"""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//bazel:local_mirror.bzl", "local_mirror")

def repo():
    maybe(
        local_mirror,
        name = "local_protoc_gen_validate",
        file_symlink = {
            "BUILD.bazel": "//third_party:local_protoc_gen_validate/src/lpm.BUILD.bazel",
            "bazel/BUILD.bazel": "//third_party:local_protoc_gen_validate/src/lpm.BUILD.bazel",
            "bazel/pgv_proto_library.bzl": "//third_party:local_protoc_gen_validate/src/bazel/pgv_proto_library.bzl",
            "validate/BUILD.bazel": "//third_party:local_protoc_gen_validate/src/validate/lpm.BUILD.bazel",
            "validate/validate.proto": "//third_party:local_protoc_gen_validate/src/validate/validate.proto",
        },
        cmake_name = "protoc_gen_validate",
        bazel_to_cmake = {},
    )
