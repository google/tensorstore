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
Local copies of bazelbuild/rules_go files.
"""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//bazel:local_mirror.bzl", "local_mirror")

def repo():
    maybe(
        local_mirror,
        name = "local_rules_go",
        file_symlink = {
            "BUILD.bazel": "//third_party:local_rules_go/src/lpm.BUILD.bazel",
            "proto/BUILD.bazel": "//third_party:local_rules_go/src/lpm.BUILD.bazel",
            "proto/def.bzl": "//third_party:local_rules_go/src/proto/def.bzl",
            "proto/compiler.bzl": "//third_party:local_rules_go/src/proto/compiler.bzl",
            "go/BUILD.bazel": "//third_party:local_rules_go/src/lpm.BUILD.bazel",
            "go/def.bzl": "//third_party:local_rules_go/src/go/def.bzl",
        },
        cmake_name = "rules_go",
        bazel_to_cmake = {},
    )
