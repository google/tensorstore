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
        name = "grpc",
        sha256 = "0af37b800953130b47c075b56683ee60bdc3eda3c37fc6004193f5b569758204",
        strip_prefix = "grpc-1.76.0",
        urls = mirror_url("https://github.com/grpc/grpc/archive/v1.76.0.tar.gz"),
        patches = [
            # Fixes -trigraph warning
            Label("//third_party:grpc/patches/fix_trigraph.diff"),
            # Fixes, including https://github.com/grpc/grpc/issues/34482
            Label("//third_party:grpc/patches/update_build_system.diff"),
            Label("//third_party:grpc/patches/pull41351.diff"),
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@com_github_cares_cares": "@c-ares",
            "@com_github_cncf_xds": "@xds",
            "@com_github_google_benchmark": "@google_benchmark",
            "@com_google_absl": "@abseil-cpp",
            "@com_google_googleapis": "@googleapis",
            "@com_google_googletest": "@googletest",
            "@com_googlesource_code_re2": "@re2",
            "@com_github_grpc_grpc": "@grpc",
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
                "--ignore-library=//bazel:python_rules.bzl",
                "--ignore-library=@rules_fuzzing//fuzzing:cc_defs.bzl",
                "--ignore-library=@build_bazel_rules_apple//apple:ios.bzl",
                "--ignore-library=@build_bazel_rules_apple//apple/testing/default_runner:ios_test_runner.bzl",
                "--ignore-library=@rules_license//rules:license.bzl",
                "--exclude-target=//:grpc_cel_engine",
                "--target=//:grpc",
                "--target=//:grpc++",
                "--target=//:grpc++_codegen_proto",
                "--target=//:grpc++_test",
                "--target=//src/compiler:grpc_cpp_plugin",
            ] + [
                "--bind=//third_party:" + k + "=" + v
                for k, v in GRPC_THIRD_PARTY_BINDINGS.items()
            ],
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
                "examples/**",
                "fuzztest/**",
                "src/core/ext/transport/binder/java/**",
            ],
        },
        cmake_target_mapping = {
            "//:grpc": "gRPC::grpc",
            "//:grpc++": "gRPC::grpc++",
            "//:grpc++_test": "gRPC::grpc_test_util",
            "//src/compiler:grpc_cpp_plugin": "gRPC::grpc_cpp_plugin",
        },
    )

# These are the set of @grpc//third_party/... aliases that are used in the
# cmake build.  It's unclear why we need to add explicit bindings for them,
# as the "repo_mapping" mechanism should catch them... but it currently does not.
# They are mainly referenced by the grpc_cc_library macros as `external_deps``.
#
# See:
# https://github.com/grpc/grpc/blob/master/bazel/grpc_build_system.bzl
# https://github.com/grpc/grpc/blob/master/third_party/BUILD
#
GRPC_THIRD_PARTY_BINDINGS = {
    "cares": "@c-ares//:ares",
    "grpc++_codegen_proto": "@grpc//:grpc++_codegen_proto",
    "grpc_cpp_plugin": "@grpc//src/compiler:grpc_cpp_plugin",
    "protobuf": "@com_google_protobuf//:protobuf",
    "protobuf_clib": "@com_google_protobuf//:protoc_lib",
    "protobuf_headers": "@com_google_protobuf//:protobuf",
    "protocol_compiler": "@com_google_protobuf//:protoc",
    "benchmark": "@com_google_benchmark//:benchmark",
    "gtest": "@googletest//:gtest",
    "libcrypto": "@boringssl//:crypto",
    "libssl": "@boringssl//:ssl",
    "libuv": "@com_github_libuv_libuv//:libuv",
    "libuv_test": "@com_github_libuv_libuv//:libuv_test",
    "madler_zlib": "@zlib//:zlib",
    "re2": "@re2//:re2",
}
