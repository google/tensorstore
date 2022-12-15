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

# NOTE: Switch back to a tagged release with darwin-arm64
# NOTE: When updating grpc, also update:
#   upb
def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_grpc_grpc",
        sha256 = "9164010d67b9080d26c2f93d4bb457231ec5a7d687dffe43d1ddf924e4b5ca6b",
        strip_prefix = "grpc-a02cc7d88ae45abf7ccb742c7c61345f7ef6d0d2",
        urls = [
            "https://github.com/grpc/grpc/archive/a02cc7d88ae45abf7ccb742c7c61345f7ef6d0d2.tar.gz",  # master(2022-11-18)
        ],
        patches = [
            "//third_party:com_github_grpc_grpc/patches/update_build_system.diff",
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@upb": "@com_google_upb",
            "@com_googlesource_code_re2": "@com_google_re2",
            "@com_github_google_benchmark": "@com_google_benchmark",
            "@com_github_cncf_udpa": "@local_proto_mirror",
            "@com_google_googleapis": "@local_proto_mirror",
        },
        cmake_name = "gRPC",
        # We currently use grpc++_test, which is not public. Fix that, test, and enable.
        cmake_enable_system_package = False,
        bazel_to_cmake = {
            "args": [
                "--ignore-library=@com_github_grpc_grpc//bazel:objc_grpc_library.bzl",
                "--ignore-library=@com_github_grpc_grpc//bazel:cython_library.bzl",
                "--ignore-library=@com_github_grpc_grpc//bazel:python_rules.bzl",
                "--ignore-library=@com_github_grpc_grpc//bazel:generate_objc.bzl",
                "--exclude-target=@com_github_grpc_grpc//:grpc_cel_engine",
                "--target=//:grpc",
                "--target=//:grpc++",
                "--target=//:grpc++_codegen_proto",
                "--target=//:grpc++_public_hdrs",
                "--target=//:grpc++_test",
                "--target=//src/compiler:grpc_cpp_plugin",
                "--bind=@com_google_protobuf//:protobuf_headers=@com_google_protobuf//:protobuf",
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
    "protobuf_headers": "@com_google_protobuf//:protobuf_headers",
    "protocol_compiler": "@com_google_protobuf//:protoc",

    # upb mappings.
    "upb_json_lib": "@com_google_upb//:json",
    "upb_lib": "@com_google_upb//:upb",
    "upb_lib_descriptor": "@com_google_upb//:descriptor_upb_proto",
    "upb_lib_descriptor_reflection": "@com_google_upb//:descriptor_upb_proto_reflection",
    "upb_reflection": "@com_google_upb//:reflection",
    "upb_textformat_lib": "@com_google_upb//:textformat",

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
