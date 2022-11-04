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

# NOTE: When updating grpc, also update:
#   com_envoyproxy_protoc_gen_validate
#   com_github_cncf_udpa
#   upb

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_grpc_grpc",
        sha256 = "13df210199d89b62cc48d6b24ba202815a3d8bf7d91c1cc38353702712b3d716",
        strip_prefix = "grpc-1.50.1",
        urls = [
            "https://github.com/grpc/grpc/archive/v1.50.1.zip",
        ],
        patches = [
            "//third_party:com_github_grpc_grpc/patches/update_build_system.diff",
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@upb": "@com_google_upb",
            "@com_googlesource_code_re2": "@com_google_re2",
            "@com_github_google_benchmark": "@com_google_benchmark",
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
