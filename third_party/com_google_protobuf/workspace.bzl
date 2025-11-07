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
load(
    "//third_party:repo.bzl",
    "mirror_url",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_protobuf",
        doc_version = "30.2",
        doc_homepage = "https://protobuf.dev/",
        strip_prefix = "protobuf-33.0",
        urls = mirror_url("https://github.com/protocolbuffers/protobuf/archive/v33.0.tar.gz"),
        sha256 = "b6b03fbaa3a90f3d4f2a3fa4ecc41d7cd0326f92fcc920a7843f12206c8d52cd",
        patches = [
            # Add mingw config properly
            Label("//third_party:com_google_protobuf/patches/build_defs_mingw.diff"),
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
                "--target=//upb/base",
                "--target=//upb/json",
                "--target=//upb/mem",
                "--target=//upb/message:compare",
                "--target=//upb/message:copy",
                "--target=//upb/message",
                "--target=//upb/port",
                "--target=//upb/reflection",
                "--target=//upb/text",
                "--target=//upb/wire",
                # upb support libraries
                "--target=//upb:generated_code_support",
                "--target=//upb/reflection:generated_reflection_support",
                "--target=//upb:upb_proto_library_copts",
                # upb plugin
                "--target=//upb_generator/c:protoc-gen-upb",
                "--target=//upb_generator/c:protoc-gen-upb_stage1",
                "--target=//upb_generator/minitable:protoc-gen-upb_minitable",
                "--target=//upb_generator/minitable:protoc-gen-upb_minitable_stage1",
                "--target=//upb_generator/reflection:protoc-gen-upbdefs",
            ] + [
                "--target=" + x
                for x in PROTO_TARGETS
            ] + [
                "--target=" + x
                for x in CMAKE_LIBRARY_ALIASES
            ],
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

# bazelisk query "//:all" | grep "_proto$"
PROTO_TARGETS = [
    "//:any_cc_proto",
    "//:any_proto",
    "//:any_upb_proto",
    "//:any_upb_reflection_proto",
    "//:api_cc_proto",
    "//:api_proto",
    "//:api_upb_proto",
    "//:api_upb_reflection_proto",
    "//:compiler_plugin_proto",
    "//:cpp_features_proto",
    "//:descriptor_proto",
    "//:duration_cc_proto",
    "//:duration_proto",
    "//:duration_upb_proto",
    "//:duration_upb_reflection_proto",
    "//:empty_cc_proto",
    "//:empty_proto",
    "//:empty_upb_proto",
    "//:empty_upb_reflection_proto",
    "//:field_mask_cc_proto",
    "//:field_mask_proto",
    # "//:field_mask_upb_proto", # alias target does not exist
    # "//:field_mask_upb_reflection_proto", # alias target does not exist
    "//:source_context_cc_proto",
    "//:source_context_proto",
    "//:source_context_upb_proto",
    "//:source_context_upb_reflection_proto",
    "//:struct_cc_proto",
    "//:struct_proto",
    "//:struct_upb_proto",
    "//:struct_upb_reflection_proto",
    "//:timestamp_cc_proto",
    "//:timestamp_proto",
    "//:timestamp_upb_proto",
    "//:type_cc_proto",
    "//:type_proto",
    "//:type_upb_proto",
    "//:type_upb_reflection_proto",
    "//:wrappers_cc_proto",
    "//:wrappers_proto",
    "//:wrappers_upb_proto",
    "//:wrappers_upb_reflection_proto",
    "//src/google/protobuf/compiler:plugin_proto",
]

# For CMake, bazel_to_cmake needs to expose the internal generated targets.
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
]

_SUFFIXES = [
    "_proto__cpp_library",
    "_proto__upb_library",
    "_proto__upbdefs_library",
    "_proto__minitable_library",
]

CMAKE_LIBRARY_ALIASES = [
    "//:" + x + s
    for x in WELL_KNOWN_TYPES + ["descriptor"]
    for s in _SUFFIXES
]
