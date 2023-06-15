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
            "imports.bzl": "//third_party:local_proto_mirror/src/imports.bzl",
            # envoyproxy/protoc-gen-validate
            "validate/validate.proto": "//third_party:local_proto_mirror/src/validate/validate.proto",
            # cncf/xds
            "xds/data/orca/v3/orca_load_report.proto": "//third_party:local_proto_mirror/src/xds/data/orca/v3/orca_load_report.proto",
            "xds/service/orca/v3/orca.proto": "//third_party:local_proto_mirror/src/xds/service/orca/v3/orca.proto",
            # protocolbuffers/protobuf
            "google/protobuf/any.proto": "//third_party:local_proto_mirror/src/google/protobuf/any.proto",
            "google/protobuf/api.proto": "//third_party:local_proto_mirror/src/google/protobuf/api.proto",
            "google/protobuf/descriptor.proto": "//third_party:local_proto_mirror/src/google/protobuf/descriptor.proto",
            "google/protobuf/duration.proto": "//third_party:local_proto_mirror/src/google/protobuf/duration.proto",
            "google/protobuf/empty.proto": "//third_party:local_proto_mirror/src/google/protobuf/empty.proto",
            "google/protobuf/field_mask.proto": "//third_party:local_proto_mirror/src/google/protobuf/field_mask.proto",
            "google/protobuf/source_context.proto": "//third_party:local_proto_mirror/src/google/protobuf/source_context.proto",
            "google/protobuf/struct.proto": "//third_party:local_proto_mirror/src/google/protobuf/struct.proto",
            "google/protobuf/timestamp.proto": "//third_party:local_proto_mirror/src/google/protobuf/timestamp.proto",
            "google/protobuf/type.proto": "//third_party:local_proto_mirror/src/google/protobuf/type.proto",
            "google/protobuf/wrappers.proto": "//third_party:local_proto_mirror/src/google/protobuf/wrappers.proto",
        },
        # Downloaded files
        file_sha256 = {},
        file_url = {},
        file_content = _BUILD_FILE_CONTENT,

        # CMake options
        cmake_name = "lpm",
        bazel_to_cmake = {},
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
# Since bazel_to_cmake doesn't have a global view of targets, it cannot apply
# aspects to collect the proto_library() sources in order to build e.g. upbdefs.
# As a workaround the upb targets are injected into the reposoitory here so that
# other upb libraries can link to the common protos.
#
#  These also become the "well-known-protos" depdendencies for upb/upbdefs.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load("@com_google_protobuf_upb//bazel:upb_proto_library.bzl",
     "upb_proto_library",
     "upb_proto_reflection_library")

proto_library(
    name = "any_proto",
    srcs = ["any.proto"],
)

proto_library(
    name = "api_proto",
    srcs = ["api.proto"],
    deps = [
        ":source_context_proto",
        ":type_proto",
    ],
)

proto_library(
    name = "duration_proto",
    srcs = ["duration.proto"],
)

proto_library(
    name = "empty_proto",
    srcs = ["empty.proto"],
)

proto_library(
    name = "field_mask_proto",
    srcs = ["field_mask.proto"],
)

proto_library(
    name = "source_context_proto",
    srcs = ["source_context.proto"],
)

proto_library(
    name = "struct_proto",
    srcs = ["struct.proto"],
)

proto_library(
    name = "timestamp_proto",
    srcs = ["timestamp.proto"],
)

proto_library(
    name = "type_proto",
    srcs = ["type.proto"],
    deps = [
        ":any_proto",
        ":source_context_proto",
    ],
)

proto_library(
    name = "wrappers_proto",
    srcs = ["wrappers.proto"],
)

proto_library(
    name = "descriptor_proto",
    srcs = ["descriptor.proto"],
)

WELL_KNOWN_TYPES = [
    "any",
    "api",
    "duration",
    "empty",
    "field_mask",
    "source_context",
    "struct",
    "timestamp",
    "type",
    "wrappers",
    "descriptor",
]

[
  upb_proto_library(
    name = x + "_upb_proto",
    deps = [":" + x + "_proto"],
  ) for x in WELL_KNOWN_TYPES
]

[
  upb_proto_reflection_library(
    name = x + "_upbdefs_proto",
    deps = [":" + x + "_proto"],
  ) for x in WELL_KNOWN_TYPES
]

# Well-known type protos

cc_library(
    name = "wkt_upb_proto",
    deps = [":" + wkt + "_upb_proto" for wkt in WELL_KNOWN_TYPES],
)

cc_library(
    name = "wkt_upbdefs_proto",
    deps = [":" + wkt + "_upbdefs_proto" for wkt in WELL_KNOWN_TYPES],
)

""",
    "BUILD.bazel": """
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["imports.bzl"])

""",
}
