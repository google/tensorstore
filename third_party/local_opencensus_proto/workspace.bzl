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
Local copies of census-instrumentation/opencensus-proto files.

From: https://github.com/census-instrumentation/opencensus-proto
"""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//bazel:local_mirror.bzl", "local_mirror")

def repo():
    maybe(
        local_mirror,
        name = "local_opencensus_proto",
        file_symlink = {
            "BUILD.bazel": "//third_party:local_opencensus_proto/src/lpm.BUILD.bazel",
            "opencensus/proto/resource/v1/BUILD.bazel": "//third_party:local_opencensus_proto/src/opencensus/proto/resource/v1/lpm.BUILD.bazel",
            "opencensus/proto/resource/v1/resource.proto": "//third_party:local_opencensus_proto/src/opencensus/proto/resource/v1/resource.proto",
            "opencensus/proto/trace/v1/BUILD.bazel": "//third_party:local_opencensus_proto/src/opencensus/proto/trace/v1/lpm.BUILD.bazel",
            "opencensus/proto/trace/v1/trace.proto": "//third_party:local_opencensus_proto/src/opencensus/proto/trace/v1/trace.proto",
            "opencensus/proto/trace/v1/trace_config.proto": "//third_party:local_opencensus_proto/src/opencensus/proto/trace/v1/trace_config.proto",
        },
        cmake_name = "opencensus_proto",
        bazel_to_cmake = {},
    )
