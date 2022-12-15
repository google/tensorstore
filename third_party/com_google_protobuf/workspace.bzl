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
        strip_prefix = "protobuf-3.21.11",
        urls = [
            "https://github.com/protocolbuffers/protobuf/releases/download/v21.11/protobuf-cpp-3.21.11.tar.gz",
        ],
        sha256 = "96f0ab99b7414e44e7bf9b218bb59510d61549ca68e648f19e3622f9999bec00",
        patches = [
            # protobuf uses rules_python, but we just use the native python rules.
            "//third_party:com_google_protobuf/patches/remove_rules_python_dependency.diff",
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@zlib": "@net_zlib",
        },
        # https://cmake.org/cmake/help/latest/module/FindProtobuf.html
        # https://github.com/protocolbuffers/protobuf/blob/master/CMakeLists.txt
        cmake_name = "Protobuf",
        cmake_extra_build_file = Label("//third_party:com_google_protobuf/cmake_extra.BUILD.bazel"),
        bazel_to_cmake = {
            "args": [
                "--target=//:protoc",
                "--target=//:protoc_lib",
                "--target=//:protobuf",
                "--target=//:protobuf_lite",
                "--target=//:protobuf_headers",
            ] + ["--target=//:" + x + "_proto" for x in _WELL_KNOWN_TYPES],
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
            ],
        },
        cmake_target_mapping = {
            "//:protobuf": "protobuf::libprotobuf",
            "//:protobuf_lite": "protobuf::libprotobuf-lite",
            "//:protoc": "protobuf::protoc",
            "//:protoc_lib": "protobuf::libprotoc",
        },
    )

_WELL_KNOWN_TYPES = [
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
    # compiler_plugin.proto is needed to build grpc and upb.
    "compiler_plugin",
]
