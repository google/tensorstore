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

# NOTE: Switch back to a tagged release with darwin-arm64
# NOTE: When updating grpc, also update:
#   upb
def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_grpc_grpc",
        sha256 = "437068b8b777d3b339da94d3498f1dc20642ac9bfa76db43abdd522186b1542b",
        strip_prefix = "grpc-1.60.0",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/grpc/grpc/archive/v1.60.0.tar.gz",
        ],
        patches = [
            # Fixes, including https://github.com/grpc/grpc/issues/34482
            Label("//third_party:com_github_grpc_grpc/patches/update_build_system.diff"),
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@upb": "@com_google_protobuf",
            "@com_googlesource_code_re2": "@com_google_re2",
            "@com_github_google_benchmark": "@com_google_benchmark",
            "@io_bazel_rules_go": "@local_proto_mirror",
        },
        cmake_name = "gRPC",
        # We currently use grpc++_test, which is not public. Fix that, test, and enable.
        cmake_enable_system_package = False,
        bazel_to_cmake = {
            "args": [
                "--ignore-library=//bazel:objc_grpc_library.bzl",
                "--ignore-library=//bazel:cython_library.bzl",
                "--ignore-library=//bazel:python_rules.bzl",
                "--ignore-library=//bazel:generate_objc.bzl",
                "--exclude-target=//:grpc_cel_engine",
                "--target=//:grpc",
                "--target=//:grpc++",
                "--target=//:grpc++_codegen_proto",
                "--target=//:grpc++_public_hdrs",
                "--target=//:grpc++_test",
                "--target=//src/compiler:grpc_cpp_plugin",
            ] + ["--bind=" + k + "=" + v for k, v in GRPC_NATIVE_BINDINGS.items()],
            "exclude": [
                "src/android/**",
                "src/csharp/**",
                "src/objective-c/**",
                "src/php/**",
                "src/ruby/**",
                "src/python/**",
                "src/proto/grpc/testing/**",
                "test/**",
                "third_party/android/**",
                "third_party/objective_c/**",
                "third_party/py/**",
                "third_party/toolchains/**",
                "third_party/upb/**",
                "tools/**",
            ],
        },
        cmake_target_mapping = {
            "//:grpc": "gRPC::grpc",
            "//:grpc++": "gRPC::grpc++",
            "//:grpc++_test": "gRPC::grpc_test_util",
            "//src/compiler:grpc_cpp_plugin": "gRPC::grpc_cpp_plugin",
        },
    )
    for key in GRPC_NATIVE_BINDINGS.keys():
        native.bind(name = key, actual = GRPC_NATIVE_BINDINGS[key])

# Aliases (unfortunately) required by gRPC.
GRPC_NATIVE_BINDINGS = {
    "cares": "@com_github_cares_cares//:ares",
    "grpc++_codegen_proto": "@com_github_grpc_grpc//:grpc++_codegen_proto",
    "grpc_cpp_plugin": "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
    "protobuf": "@com_google_protobuf//:protobuf",
    "protobuf_clib": "@com_google_protobuf//:protoc_lib",
    "protobuf_headers": "@com_google_protobuf//:protobuf",
    "protocol_compiler": "@com_google_protobuf//:protoc",

    # upb mappings.
    "upb_json_lib": "@com_google_protobuf//upb:json",
    "upb_lib": "@com_google_protobuf//upb:upb",
    "upb_lib_descriptor": "@com_google_protobuf//upb:cmake_descriptor_upb",
    "upb_lib_descriptor_reflection": "@com_google_protobuf//upb:cmake_descriptor_upbdefs",
    "upb_reflection": "@com_google_protobuf//upb:reflection",
    "upb_textformat_lib": "@com_google_protobuf//upb:text",
    "upb_collections_lib": "@com_google_protobuf//upb:collections",
    "upb_base_lib": "@com_google_protobuf//upb:base",
    "upb_mem_lib": "@com_google_protobuf//upb:mem",

    # These exist to be used by grpc_build_system.bzl
    "benchmark": "@com_google_benchmark//:benchmark",
    "gtest": "@com_google_googletest//:gtest",
    "libcrypto": "@com_google_boringssl//:crypto",
    "libssl": "@com_google_boringssl//:ssl",
    "libuv": "@com_github_libuv_libuv//:libuv",
    "libuv_test": "@com_github_libuv_libuv//:libuv_test",
    "madler_zlib": "@net_zlib//:zlib",
    "re2": "@com_google_re2//:re2",
}
