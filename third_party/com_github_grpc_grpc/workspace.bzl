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
        name = "com_github_grpc_grpc",
        sha256 = "a49e6ed0ef16a4b12fefad44e7aec6f6cd3843d5a0a3ba66709565b74a62f595",
        strip_prefix = "grpc-1.46.3",
        urls = [
            "https://github.com/grpc/grpc/archive/v1.46.3.zip",
        ],
        repo_mapping = {
            "@upb": "@com_google_upb",
        },
    )

    # Aliases (unfortunately) required by gRPC
    native.bind(
        name = "upb_lib",
        actual = "@com_google_upb//:upb",
    )

    native.bind(
        name = "upb_lib_descriptor",
        actual = "@com_google_upb//:descriptor_upb_proto",
    )

    native.bind(
        name = "upb_lib_descriptor_reflection",
        actual = "@com_google_upb//:descriptor_upb_proto_reflection",
    )

    native.bind(
        name = "upb_textformat_lib",
        actual = "@com_google_upb//:textformat",
    )

    native.bind(
        name = "upb_json_lib",
        actual = "@com_google_upb//:json",
    )

    native.bind(
        name = "protocol_compiler",
        actual = "@com_google_protobuf//:protoc",
    )

    native.bind(
        name = "absl",
        actual = "@com_google_absl//absl",
    )

    native.bind(
        name = "absl-base",
        actual = "@com_google_absl//absl/base",
    )

    native.bind(
        name = "absl-time",
        actual = "@com_google_absl//absl/time:time",
    )

    native.bind(
        name = "libssl",
        actual = "@com_google_boringssl//:ssl",
    )

    native.bind(
        name = "madler_zlib",
        actual = "@net_zlib//:zlib",
    )

    native.bind(
        name = "protobuf",
        actual = "@com_google_protobuf//:protobuf",
    )

    native.bind(
        name = "protobuf_clib",
        actual = "@com_google_protobuf//:protoc_lib",
    )

    native.bind(
        name = "protobuf_headers",
        actual = "@com_google_protobuf//:protobuf_headers",
    )

    native.bind(
        name = "re2",
        actual = "@com_google_re2//:re2",
    )

    native.bind(
        name = "cares",
        actual = "@com_github_cares_cares//:ares",
    )
