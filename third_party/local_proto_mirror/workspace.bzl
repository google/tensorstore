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

From: https://github.com/envoyproxy/protoc-gen-validate

    validate/validate.proto
    bazel/pgv_proto_library.bzl

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

From: https://github.com/bazelbuild/rules_go/

    proto/def.bzl
    go/def.bzl


From: https://github.com/census-instrumentation/opencensus-proto

    opencensus/proto/resource/v1/resource.proto
    opencensus/proto/trace/v1/trace.proto
    opencensus/proto/trace/v1/trace_config.proto

"""

load("//bazel:local_mirror.bzl", "local_mirror")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        local_mirror,
        name = "local_proto_mirror",
        # Local files. Avoid BUILD.bazel files as they will be picked up by the //... pattern.
        file_symlink = {
            "BUILD.bazel": "//third_party:local_proto_mirror/src/lpm.BUILD.bazel",
            "imports.bzl": "//third_party:local_proto_mirror/src/imports.bzl",
            # bazelbuild/rules_go
            "proto/BUILD.bazel": "//third_party:local_proto_mirror/src/lpm.BUILD.bazel",
            "proto/def.bzl": "//third_party:local_proto_mirror/src/proto/def.bzl",
            "go/BUILD.bazel": "//third_party:local_proto_mirror/src/lpm.BUILD.bazel",
            "go/def.bzl": "//third_party:local_proto_mirror/src/go/def.bzl",
            # census-instrumentation/opencensus-proto
            "opencensus/proto/resource/v1/BUILD.bazel": "//third_party:local_proto_mirror/src/opencensus/proto/resource/v1/lpm.BUILD.bazel",
            "opencensus/proto/resource/v1/resource.proto": "//third_party:local_proto_mirror/src/opencensus/proto/resource/v1/resource.proto",
            "opencensus/proto/trace/v1/BUILD.bazel": "//third_party:local_proto_mirror/src/opencensus/proto/trace/v1/lpm.BUILD.bazel",
            "opencensus/proto/trace/v1/trace.proto": "//third_party:local_proto_mirror/src/opencensus/proto/trace/v1/trace.proto",
            "opencensus/proto/trace/v1/trace_config.proto": "//third_party:local_proto_mirror/src/opencensus/proto/trace/v1/trace_config.proto",
            # envoyproxy/protoc-gen-validate
            "bazel/BUILD.bazel": "//third_party:local_proto_mirror/src/lpm.BUILD.bazel",
            "bazel/pgv_proto_library.bzl": "//third_party:local_proto_mirror/src/bazel/pgv_proto_library.bzl",
            "validate/BUILD.bazel": "//third_party:local_proto_mirror/src/validate/lpm.BUILD.bazel",
            "validate/validate.proto": "//third_party:local_proto_mirror/src/validate/validate.proto",
            # protocolbuffers/protobuf
            "google/protobuf/BUILD.bazel": "//third_party:local_proto_mirror/src/google/protobuf/lpm.BUILD.bazel",
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
        file_content = {},

        # CMake options
        cmake_name = "lpm",
        bazel_to_cmake = {},
    )
