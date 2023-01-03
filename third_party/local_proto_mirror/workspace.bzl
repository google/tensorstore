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
These are here to support, for example, building upb code from cmake.

From: https://github.com/cncf/xds

   xds/service/orca/v3:orca.proto (null)
   xds/data/orca/v3:orca_load_report.proto (null)

From: https://github.com/envoyproxy/protoc-gen-validate

    validate/validate.proto

From: https://github.com/googleapis/googleapis

    google/rpc/code.proto
    google/rpc/error_details.proto
    google/rpc/status.proto


From: https://github.com/protocolbuffers/protobuf

    google/protobuf/any.proto
    google/protobuf/api.proto
    google/protobuf/descriptor.proto
    google/protobuf/duration.proto
    google/protobuf/empty.proto
    google/protobuf/field_mask.proto
    google/protobuf/source_context.proto
    google/protobuf/struct.proto
    google/protobuf/timestamp.proto
    google/protobuf/type.proto
    google/protobuf/wrappers.proto

"""

load("//bazel:local_mirror.bzl", "local_mirror")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        local_mirror,
        name = "local_proto_mirror",
        files = [
            "google/protobuf/BUILD.bazel",
            "google/protobuf/any.proto",
            "google/protobuf/api.proto",
            "google/protobuf/descriptor.proto",
            "google/protobuf/duration.proto",
            "google/protobuf/empty.proto",
            "google/protobuf/field_mask.proto",
            "google/protobuf/source_context.proto",
            "google/protobuf/struct.proto",
            "google/protobuf/timestamp.proto",
            "google/protobuf/type.proto",
            "google/protobuf/wrappers.proto",
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
        file_sha256 = {
            "google/protobuf/any.proto": "17eeb5dc4a65300beac9ca6d7ebb6bfabfd1825fec2338013687a5805293e606",
            "google/protobuf/api.proto": "9daa858a1c529f47f9d4ed1bc2df0dab6154ad42f99cf0351b3da9d4daff8eb0",
            "google/protobuf/descriptor.proto": "bf4da2988d089b0454ecceb43d07c82085fc419ac2436384fa543b7b1691277c",
            "google/protobuf/duration.proto": "a3f7301ff2956ec2e30c2241ece07197e4a86c752348d5607224819d4921c9fe",
            "google/protobuf/empty.proto": "ecef3d54cc9e079673b9816c67bac770f7f3bf6dada2d4596ba69d71daa971e6",
            "google/protobuf/field_mask.proto": "17b0f342cc3a262e317d56884f3f575afbfac512ad0579349cc7450e41f26891",
            "google/protobuf/source_context.proto": "37d3401de89af1d1496fc949924a5f2011bd2fadfbeb0da1caa277159a96776d",
            "google/protobuf/struct.proto": "dd2b492d6683d52eec6eb2bdb3cce7e2cee126006eab94060fff513d58dcb62b",
            "google/protobuf/timestamp.proto": "fd8d738e0cb5301455a957b16a984d2c30c7ac377850ca72dbfa0e8c0f27ec15",
            "google/protobuf/type.proto": "59b9e8322f547748f137b02bd570404851cc220c8ff4963f76a5217cf84e3bcf",
            "google/protobuf/wrappers.proto": "a26c1d6ec73a592ac31796289a61ceffe2db2453ff427dfdd7697aac50844280",
            "google/rpc/code.proto": "55ba84362212a7440830330d47279a78549a5cd9a7632017c7acc796ea5aa18c",
            "google/rpc/error_details.proto": "41c6ea79dd50d433f31d96623e0b3fa83736a797434570445fa3e65bb7f62eb3",
            "google/rpc/status.proto": "108562c2d7854812908cdcd0d988132880822b05ab5e4f28960e1a5572217854",
            "validate/validate.proto": "bf7ca2ac45a75b8b9ff12f38efd7f48ee460ede1a7919d60c93fad3a64fc2eee",
            "xds/data/orca/v3/orca_load_report.proto": "8721df1147c6094b61f9df4becd6a0300c3378e6f50be60d1d5f2e4860769264",
            "xds/service/orca/v3/orca.proto": "90150a0294c560212990d61a23cb0f0aec8033101cb9ca00f2fdaf8dccc8489b",
        },
        file_url = {
            # https://github.com/protocolbuffers/protobuf
            "google/protobuf/any.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/any.proto",
            ],
            "google/protobuf/api.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/api.proto",
            ],
            "google/protobuf/descriptor.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/descriptor.proto",
            ],
            "google/protobuf/duration.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/duration.proto",
            ],
            "google/protobuf/empty.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/empty.proto",
            ],
            "google/protobuf/field_mask.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/field_mask.proto",
            ],
            "google/protobuf/source_context.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/source_context.proto",
            ],
            "google/protobuf/struct.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/struct.proto",
            ],
            "google/protobuf/timestamp.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/timestamp.proto",
            ],
            "google/protobuf/type.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/type.proto",
            ],
            "google/protobuf/wrappers.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/protocolbuffers/protobuf/60b71498d70a5645324385269c518b95c8c2feb0/src/google/protobuf/wrappers.proto",
            ],
            # https://github.com/googleapis/googleapis
            "google/rpc/code.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/googleapis/googleapis/83c3605afb5a39952bf0a0809875d41cf2a558ca/google/rpc/code.proto",
            ],
            "google/rpc/error_details.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/googleapis/googleapis/83c3605afb5a39952bf0a0809875d41cf2a558ca/google/rpc/error_details.proto",
            ],
            "google/rpc/status.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/googleapis/googleapis/83c3605afb5a39952bf0a0809875d41cf2a558ca/google/rpc/status.proto",
            ],
            # https://github.com/cncf/xds
            "xds/service/orca/v3/orca.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/cncf/xds/1e77728a1eaa11d6c931ec2ccd6e95f516a7ef94/xds/service/orca/v3/orca.proto",
            ],
            "xds/data/orca/v3/orca_load_report.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/cncf/xds/1e77728a1eaa11d6c931ec2ccd6e95f516a7ef94/xds/data/orca/v3/orca_load_report.proto",
            ],
            # https://github.com/envoyproxy/protoc-gen-validate
            "validate/validate.proto": [
                "https://storage.googleapis.com/tensorstore-bazel-mirror/raw.githubusercontent.com/bufbuild/protoc-gen-validate/2682ad06cca00550030e177834f58a2bc06eb61e/validate/validate.proto",
            ],
        },
        file_content = _BUILD_FILE_CONTENT,
        cmake_name = "lpm",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//google/rpc:code_proto": "lpm::google_rpc_code_proto",
            "//google/rpc:code_upb_proto": "lpm::google_rpc_code_upb_proto",
            "//google/rpc:code_upbdef_proto": "lpm::google_rpc_code_upbdef_proto",
            "//google/rpc:error_details_cc_proto": "lpm::google_rpc_error_details_cc_proto",
            "//google/rpc:error_details_proto": "lpm::google_rpc_error_details_proto",
            "//google/rpc:error_details_upb_proto": "lpm::google_rpc_error_details_upb_proto",
            "//google/rpc:error_details_upbdef_proto": "lpm::google_rpc_error_details_upbdef_proto",
            "//google/rpc:status_cc_proto": "lpm::google_rpc_status_cc_proto",
            "//google/rpc:status_proto": "lpm::google_rpc_status_proto",
            "//google/rpc:status_upb_proto": "lpm::google_rpc_status_upb_proto",
            "//google/rpc:status_upbdef_proto": "lpm::google_rpc_status_upbdef_proto",
            "//validate:validate_proto": "lpm::validate_proto",
            "//validate:validate_proto_upb": "lpm::validate_proto_upb",
            "//validate:validate_proto_upbdef": "lpm::validate_proto_upbdef",
            "//xds/data/orca/v3:pkg": "lpm::xds_data_orca_v3_pkg",
            "//xds/data/orca/v3:pkg_upb": "lpm::xds_data_orca_v3_pkg_upb",
            "//xds/data/orca/v3:pkg_upbdef": "lpm::xds_data_orca_v3_pkg_upbdef",
            "//xds/service/orca/v3:pkg": "lpm::xds_service_orca_v3_pkg",
            "//xds/service/orca/v3:pkg_upb": "lpm::xds_service_orca_v3_pkg_upb",
            "//xds/service/orca/v3:pkg_upbdef": "lpm::xds_service_orca_v3_pkg_upbdef",
        },
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

load("@com_google_upb//bazel:upb_proto_library.bzl", "upb_proto_library", "upb_proto_reflection_library")

upb_proto_library(name = "validate_proto_upb", deps = [":validate_proto"])
upb_proto_reflection_library(name = "validate_proto_upbdef", deps = [":validate_proto"])

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

load("@com_google_upb//bazel:upb_proto_library.bzl", "upb_proto_library", "upb_proto_reflection_library")

upb_proto_library(name = "pkg_upb", deps = [":pkg"])
upb_proto_reflection_library(name = "pkg_upbdef", deps = [":pkg"])

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

load("@com_google_upb//bazel:upb_proto_library.bzl", "upb_proto_library", "upb_proto_reflection_library")

upb_proto_library(name = "pkg_upb", deps = [":pkg"])
upb_proto_reflection_library(name = "pkg_upbdef", deps = [":pkg"])

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

load("@com_google_upb//bazel:upb_proto_library.bzl", "upb_proto_library", "upb_proto_reflection_library")

upb_proto_library(
    name = "code_upb_proto",
    deps = [":code_proto"],
)

upb_proto_library(
    name = "error_details_upb_proto",
    deps = [":error_details_proto"],
)

upb_proto_library(
    name = "status_upb_proto",
    deps = [":status_proto"],
)

upb_proto_reflection_library(
    name = "code_upbdef_proto",
    deps = [":code_proto"],
)

upb_proto_reflection_library(
    name = "error_details_upbdef_proto",
    deps = [":error_details_proto"],
)

upb_proto_reflection_library(
    name = "status_upbdef_proto",
    deps = [":status_proto"],
)

""",
    "google/protobuf/BUILD.bazel": """

package(default_visibility = ["//visibility:public"])


# Since bazel_to_cmake doesn't have a global view of targets, it cannot apply
# aspects to collect the proto_library() sources in order to build e.g. upbdefs.
# As a workaround the upb targets are injected into the reposoitory here so that
# other upb libraries can link to the common protos.
#
#  These also become the "well-known-protos" depdendencies for upb/upbdefs.

load("@com_google_upb//bazel:upb_proto_library.bzl", "upb_proto_library", "upb_proto_reflection_library")

proto_library(
    name = "well_known_protos",
    srcs = [
        "any.proto",
        "api.proto",
        "descriptor.proto",
        "duration.proto",
        "empty.proto",
        "field_mask.proto",
        "source_context.proto",
        "struct.proto",
        "timestamp.proto",
        "type.proto",
        "wrappers.proto",
    ],
)

upb_proto_library(
    name = "well_known_protos_upb",
    deps = [ ":well_known_protos" ],
)

upb_proto_reflection_library(
    name = "well_known_protos_upbdefs",
    deps = [ ":well_known_protos" ],
)

""",
}
