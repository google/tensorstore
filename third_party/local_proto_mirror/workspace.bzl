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

From: https://github.com/googleapis/googleapis

    google/rpc/code.proto
    google/rpc/error_details.proto
    google/rpc/status.proto

"""

load("//bazel:local_mirror.bzl", "local_mirror")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        local_mirror,
        name = "local_proto_mirror",
        files = [
            "google/rpc/BUILD.bazel",
            "google/rpc/code.proto",
            "google/rpc/error_details.proto",
            "google/rpc/status.proto",
            "validate/BUILD.bazel",
            "validate/validate.proto",
            "xds/data/orca/v3/BUILD.bazel",
            "xds/data/orca/v3/orca_load_report.proto",
            "xds/service/orca/v3/BUILD.bazel",
            "xds/service/orca/v3/orca.proto",
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
            "google/rpc/code.proto": [
                "https://raw.githubusercontent.com/googleapis/googleapis/83c3605afb5a39952bf0a0809875d41cf2a558ca/google/rpc/code.proto",
            ],
            "google/rpc/error_details.proto": [
                "https://raw.githubusercontent.com/googleapis/googleapis/83c3605afb5a39952bf0a0809875d41cf2a558ca/google/rpc/error_details.proto",
            ],
            "google/rpc/status.proto": [
                "https://raw.githubusercontent.com/googleapis/googleapis/83c3605afb5a39952bf0a0809875d41cf2a558ca/google/rpc/status.proto",
            ],
        },
        file_sha256 = {
            "google/rpc/code.proto": "55ba84362212a7440830330d47279a78549a5cd9a7632017c7acc796ea5aa18c",
            "google/rpc/error_details.proto": "41c6ea79dd50d433f31d96623e0b3fa83736a797434570445fa3e65bb7f62eb3",
            "google/rpc/status.proto": "108562c2d7854812908cdcd0d988132880822b05ab5e4f28960e1a5572217854",
            "validate/validate.proto": "bf7ca2ac45a75b8b9ff12f38efd7f48ee460ede1a7919d60c93fad3a64fc2eee",
            "xds/data/orca/v3/orca_load_report.proto": "8721df1147c6094b61f9df4becd6a0300c3378e6f50be60d1d5f2e4860769264",
            "xds/service/orca/v3/orca.proto": "90150a0294c560212990d61a23cb0f0aec8033101cb9ca00f2fdaf8dccc8489b",
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
    "google/rpc/BUILD.bazel": """
package(default_visibility = ["//visibility:public"])

proto_library(
    name = "code_proto",
    srcs = ["code.proto"],
    deps = [],
)

proto_library(
    name = "error_details_proto",
    srcs = ["error_details.proto"],
    deps = [
        "@com_google_protobuf//:duration_proto",
    ],
)

proto_library(
    name = "status_proto",
    srcs = ["status.proto"],
    deps = [
        "@com_google_protobuf//:any_proto",
    ],
)

cc_proto_library(
    name = "code_cc_proto",
    deps = [":code_proto"],
)

cc_proto_library(
    name = "error_details_cc_proto",
    deps = [":error_details_proto"],
)

cc_proto_library(
    name = "status_cc_proto",
    deps = [":status_proto"],
)

""",
}
