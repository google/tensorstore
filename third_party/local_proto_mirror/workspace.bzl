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
        # Local files. Avoid BUILD.bazel files as they will be picked up by the //... pattern.
        file_symlink = {
            "validate/validate.proto": "//third_party:local_proto_mirror/src/validate/validate.proto",
            "xds/data/orca/v3/orca_load_report.proto": "//third_party:local_proto_mirror/src/xds/data/orca/v3/orca_load_report.proto",
            "xds/service/orca/v3/orca.proto": "//third_party:local_proto_mirror/src/xds/service/orca/v3/orca.proto",
            "imports.bzl": "//third_party:local_proto_mirror/src/imports.bzl",
        },  # Map source Label() -> repo path
        # Downloaded files
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
        },
        file_content = _BUILD_FILE_CONTENT,
        # CMake options
        cmake_name = "lpm",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//google/protobuf:descriptor_proto": "lpm::google_protobuf_descriptor_proto",
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

load("@com_google_protobuf_upb//bazel:upb_proto_library.bzl",
     "upb_proto_library",
     "upb_proto_reflection_library")

upb_proto_library(
    name = "validate_proto_upb",
    deps = [":validate_proto"],
)

upb_proto_reflection_library(
    name = "validate_proto_upbdef",
    deps = [":validate_proto"],
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

load("@com_google_protobuf_upb//bazel:upb_proto_library.bzl",
     "upb_proto_library",
     "upb_proto_reflection_library")

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

load("@com_google_protobuf_upb//bazel:upb_proto_library.bzl",
     "upb_proto_library",
     "upb_proto_reflection_library")

upb_proto_library(name = "pkg_upb", deps = [":pkg"])
upb_proto_reflection_library(name = "pkg_upbdef", deps = [":pkg"])

""",
    "google/protobuf/BUILD.bazel": """

package(default_visibility = ["//visibility:public"])


# Since bazel_to_cmake doesn't have a global view of targets, it cannot apply
# aspects to collect the proto_library() sources in order to build e.g. upbdefs.
# As a workaround the upb targets are injected into the reposoitory here so that
# other upb libraries can link to the common protos.
#
#  These also become the "well-known-protos" depdendencies for upb/upbdefs.

load("@com_google_protobuf_upb//bazel:upb_proto_library.bzl",
     "upb_proto_library",
     "upb_proto_reflection_library")

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
    "BUILD.bazel": """
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["imports.bzl"])

""",
}
