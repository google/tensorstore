# Copyright 2022 The TensorStore Authors
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
Local copies of protocol buffers used by gRPC and other code.

From: https://github.com/cncf/xds

   xds/service/orca/v3:orca.proto (null)
   xds/data/orca/v3:orca_load_report.proto (null)

From: https://github.com/envoyproxy/protoc-gen-validate

    validate/validate.proto

"""

load("//bazel:local_mirror.bzl", "local_mirror")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        local_mirror,
        name = "local_proto_mirror",
        files = [
            "xds/service/orca/v3/orca.proto",
            "xds/data/orca/v3/orca_load_report.proto",
            "validate/validate.proto",
            "xds/service/orca/v3/BUILD.bazel",
            "xds/data/orca/v3/BUILD.bazel",
            "validate/BUILD.bazel",
        ],
        file_url = {
            "xds/service/orca/v3/orca.proto": [
                "https://raw.githubusercontent.com/cncf/xds/1e77728a1eaa11d6c931ec2ccd6e95f516a7ef94/xds/service/orca/v3/orca.proto",
            ],
            "xds/data/orca/v3/orca_load_report.proto": [
                "https://raw.githubusercontent.com/cncf/xds/1e77728a1eaa11d6c931ec2ccd6e95f516a7ef94/xds/data/orca/v3/orca_load_report.proto",
            ],
            "validate/validate.proto": [
                "https://raw.githubusercontent.com/bufbuild/protoc-gen-validate/2682ad06cca00550030e177834f58a2bc06eb61e/validate/validate.proto",
            ],
        },
        file_sha256 = {
            "xds/service/orca/v3/orca.proto": "90150a0294c560212990d61a23cb0f0aec8033101cb9ca00f2fdaf8dccc8489b",
            "xds/data/orca/v3/orca_load_report.proto": "8721df1147c6094b61f9df4becd6a0300c3378e6f50be60d1d5f2e4860769264",
            "validate/validate.proto": "bf7ca2ac45a75b8b9ff12f38efd7f48ee460ede1a7919d60c93fad3a64fc2eee",
        },
        file_content = _BUILD_FILE_CONTENT,
    )

# Mapping from file name to build file content.
_BUILD_FILE_CONTENT = {
    "validate/BUILD.bazel": """
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

proto_library(
    name = "validate_proto",
    srcs = ["validate.proto"],
    deps = [
        "@com_google_protobuf//:descriptor_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
)
""",
    "xds/data/orca/v3/BUILD.bazel": """
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

proto_library(
    name = "pkg",
    srcs = ["orca_load_report.proto"],
    deps = [
        "//validate:validate_proto",
    ],
)
""",
    "xds/service/orca/v3/BUILD.bazel": """
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

proto_library(
    name = "pkg",
    srcs = ["orca.proto"],
    deps = [
        "//validate:validate_proto",
        "//xds/data/orca/v3:pkg",
        "@com_google_protobuf//:duration_proto",
    ],
)
""",
}
