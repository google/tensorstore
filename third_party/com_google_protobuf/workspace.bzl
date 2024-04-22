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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_protobuf",
        strip_prefix = "protobuf-25.1",
        sha256 = "9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/protocolbuffers/protobuf/archive/v25.1.tar.gz",
            "https://github.com/protocolbuffers/protobuf/archive/v25.1.tar.gz",  # 25.x(2023-06-13)
        ],
        patches = [
            # protobuf uses rules_python, but we just use the native python rules.
            Label("//third_party:com_google_protobuf/patches/remove_rules_python_dependency.diff"),
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@zlib": "@net_zlib",
            "@utf8_range": "@com_google_protobuf_utf8_range",
        },
        # https://cmake.org/cmake/help/latest/module/FindProtobuf.html
        # https://github.com/protocolbuffers/protobuf/blob/master/CMakeLists.txt
        cmake_name = "protobuf",
        cmake_extra_build_file = Label("//third_party:com_google_protobuf/cmake_extra.BUILD.bazel"),
        bazel_to_cmake = {
            "args": [
                # required by bazel_to_cmake
                "--bind=//src/google/protobuf:wkt_cc_proto=//src/google/protobuf:cmake_wkt_cc_proto",
                # Ignore libraries
                "--ignore-library=//bazel:amalgamation.bzl",
                "--ignore-library=//bazel:py_proto_library.bzl",
                "--ignore-library=//bazel:python_downloads.bzl",
                "--ignore-library=//bazel:system_python.bzl",
                "--ignore-library=//bazel:workspace_deps.bzl",
                "--ignore-library=//bazel:upb_proto_library_internal/aspect.bzl",
                "--ignore-library=//bazel:upb_proto_library_internal/cc_library_func.bzl",
                "--ignore-library=//protos/bazel:upb_cc_proto_library.bzl",
                "--ignore-library=//python/dist:dist.bzl",
                "--ignore-library=//python:py_extension.bzl",
                "--ignore-library=//benchmarks:build_defs.bzl",
                "--ignore-library=@rules_python//python:defs.bzl",
                "--target=//:protobuf",
                "--target=//:protobuf_lite",
                "--target=//:protoc",
                "--target=//:protoc_lib",
                "--target=//src/google/protobuf:protobuf",
                "--target=//src/google/protobuf:protobuf_lite",
                "--target=//src/google/protobuf/compiler:protoc_lib",
                "--target=//src/google/protobuf/compiler:code_generator",
                "--target=//:descriptor_proto_srcs",
                # upb
                "--target=//upb:collections",
                "--target=//upb:descriptor_upb_proto_reflection",
                "--target=//upb:descriptor_upb_proto",
                "--target=//upb:json",
                "--target=//upb:message",
                "--target=//upb:port",
                "--target=//upb:reflection",
                "--target=//upb:text",
                "--target=//upb:upb",
                "--target=//upb:base",
                "--target=//upb:mem",
                # upb support libraries
                "--target=//upb:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
                "--target=//upb:generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
                "--target=//upb:upb_proto_library_copts__for_generated_code_only_do_not_use",
                # upb plugin
                "--target=//upb_generator:protoc-gen-upb_stage1",
                "--target=//upb_generator:protoc-gen-upb",
                "--target=//upb_generator:protoc-gen-upbdefs",
                "--target=//upb_generator:protoc-gen-upb_minitable_stage1",
            ] + EXTRA_PROTO_TARGETS,
            "exclude": [
                "benchmarks/**",
                "ci/**",
                "cmake/**",
                "conformance/**",
                "docs/**",
                "editors/**",
                "examples/**",
                "kokoro/**",
                "pkg/**",
                "toolchain/**",
                "upb/cmake/**",
                # Disable languages
                "csharp/**",
                "java/**",
                "lua/**",
                "objectivec/**",
                "php/**",
                "ruby/**",
                "rust/**",
                "python/**",
                "protos/**",  # future C++ api?
            ],
        },
        cmake_target_mapping = PROTOBUF_CMAKE_MAPPING,
        cmakelists_prefix = CMAKELISTS_PREFIX,
    )

CMAKELISTS_PREFIX = """
set(Protobuf_IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")
set(Protobuf_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")
set(Protobuf_LIBRARIES "protobuf::libprotobuf" CACHE INTERNAL "")
"""

PROTOBUF_CMAKE_MAPPING = {
    "//:protobuf": "protobuf::libprotobuf",
    "//:protobuf_lite": "protobuf::libprotobuf-lite",
    "//:protoc": "protobuf::protoc",
    "//:protoc_lib": "protobuf::libprotoc",
}

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
    # add descriptor
    "descriptor",
]

SUFFIXES = [
    "",
    "__cpp_library",
    "__upb_library",
    "__upbdefs_library",
    "__minitable_library",
]

EXTRA_PROTO_TARGETS = [
    "--target=//:" + x + "_proto" + y
    for x in WELL_KNOWN_TYPES
    for y in SUFFIXES
]
