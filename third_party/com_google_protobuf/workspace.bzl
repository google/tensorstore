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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_protobuf",
        doc_version = "30.2",
        doc_homepage = "https://protobuf.dev/",
        strip_prefix = "protobuf-30.2",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/protocolbuffers/protobuf/archive/v30.2.tar.gz",
        ],
        sha256 = "07a43d88fe5a38e434c7f94129cad56a4c43a51f99336074d0799c2f7d4e44c5",
        patches = [
            # protobuf uses rules_java, but we don't want to import it.
            Label("//third_party:com_google_protobuf/patches/no_rules_java.diff"),
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@com_google_absl": "@abseil-cpp",
            "@proto_bazel_features": "@bazel_features",
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
                "--ignore-library=//bazel/private:native.bzl",
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
                "--ignore-library=//src/google/protobuf/editions:defaults.bzl",
                "--ignore-library=//bazel/private:bazel_cc_proto_library.bzl",
                "--ignore-library=//bazel/private:java_lite_proto_library.bzl",
                "--ignore-library=//bazel/private:proto_toolchain_rule.bzl",
                "--ignore-library=@bazel_skylib//:bzl_library.bzl",
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
                "--target=//upb:base",
                "--target=//upb:json",
                "--target=//upb:mem",
                "--target=//upb:message_compare",
                "--target=//upb:message_copy",
                "--target=//upb:message",
                "--target=//upb:port",
                "--target=//upb:reflection",
                "--target=//upb:text",
                "--target=//upb:wire",
                # upb support libraries
                "--target=//upb:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
                "--target=//upb:generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
                "--target=//upb:upb_proto_library_copts__for_generated_code_only_do_not_use",
                # upb plugin
                "--target=//upb_generator/c:protoc-gen-upb",
                "--target=//upb_generator/c:protoc-gen-upb_stage1",
                "--target=//upb_generator/minitable:protoc-gen-upb_minitable",
                "--target=//upb_generator/minitable:protoc-gen-upb_minitable_stage1",
                "--target=//upb_generator/reflection:protoc-gen-upbdefs",
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
                "hpb/**",
                "java/**",
                "lua/**",
                "objectivec/**",
                "php/**",
                "python/**",
                "ruby/**",
                "rust/**",
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
