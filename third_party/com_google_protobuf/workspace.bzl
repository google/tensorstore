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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_protobuf",
        strip_prefix = "protobuf-58b6ddb03ef8f186c9225d0107213f74750a82f3",
        sha256 = "9ff1badbc558c17bfcbda54dbb183338331cd77afa0c47e584f512d71f1f6e80",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/protocolbuffers/protobuf/archive/58b6ddb03ef8f186c9225d0107213f74750a82f3.tar.gz",
            "https://github.com/protocolbuffers/protobuf/archive/58b6ddb03ef8f186c9225d0107213f74750a82f3.tar.gz",  # 23.x(2023-06-13)
        ],
        patches = [
            # protobuf uses rules_python, but we just use the native python rules.
            Label("//third_party:com_google_protobuf/patches/remove_rules_python_dependency.diff"),
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@zlib": "@net_zlib",
            "@upb": "@com_google_protobuf_upb",
            "@utf8_range": "@com_google_protobuf_utf8_range",
        },
        # https://cmake.org/cmake/help/latest/module/FindProtobuf.html
        # https://github.com/protocolbuffers/protobuf/blob/master/CMakeLists.txt
        cmake_name = "Protobuf",
        cmake_extra_build_file = Label("//third_party:com_google_protobuf/cmake_extra.BUILD.bazel"),
        bazel_to_cmake = {
            "args": [
                "--ignore-library=@upb//bazel:workspace_deps.bzl",
                "--ignore-library=@upb//bazel:system_python.bzl",
                "--bind=//src/google/protobuf:wkt_cc_proto=//:b2cmake_wkt_cc_proto",
                # required by bazel_to_cmake
                "--target=//:b2cmake_wkt_cc_proto",
                "--target=//:protobuf",
                "--target=//:protobuf_lite",
                "--target=//:protoc",
                "--target=//:protoc_lib",
                "--target=//src/google/protobuf:protobuf",
                "--target=//src/google/protobuf:protobuf_lite",
                "--target=//src/google/protobuf/compiler:protoc_lib",
                "--target=//src/google/protobuf/compiler:code_generator",
                "--target=//:descriptor_proto_srcs",
                "--target=//:compiler_plugin_proto",
            ] + EXTRA_PROTO_TARGETS,
            "exclude": [
                "cmake/**",
                "conformance/**",
                "docs/**",
                "editors/**",
                "examples/**",
                "kokoro/**",
                "pkg/**",
                "toolchain/**",
                # Disable languages
                "csharp/**",
                "java/**",
                "objectivec/**",
                "php/**",
                "ruby/**",
                "rust/**",
            ],
        },
        cmake_target_mapping = {
            "//:protobuf": "protobuf::libprotobuf",
            "//:protobuf_lite": "protobuf::libprotobuf-lite",
            "//:protoc": "protobuf::protoc",
            "//:protoc_lib": "protobuf::libprotoc",
        },
        cmakelists_prefix = """
set(Protobuf_IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")
set(Protobuf_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")
set(Protobuf_LIBRARIES "protobuf::libprotobuf" CACHE INTERNAL "")
""",
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
    # Descriptor.proto isn't considered "well known", but is available via
    # :protobuf and :protobuf_wkt
    "descriptor",
]

EXTRA_PROTO_TARGETS = [
    "--target=//:" + x + "_proto"
    for x in WELL_KNOWN_TYPES
] + [
    "--target=//src/google/protobuf:" + x + "_upb_proto"
    for x in WELL_KNOWN_TYPES
] + [
    "--target=//src/google/protobuf:" + x + "_upb_proto_reflection"
    for x in WELL_KNOWN_TYPES
]
