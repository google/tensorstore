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

# Must match grpc upb, see grpc/bazel/grpc_deps.bzl

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_protobuf_upb",
        sha256 = "ee29f1e674dfb3d0121937c6256771ccaafa3ce75410a6129880bd9c6ee2b546",
        strip_prefix = "upb-1956df14832979471d0c79210a817aeb54f7a526",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/protocolbuffers/upb/archive/1956df14832979471d0c79210a817aeb54f7a526.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/1956df14832979471d0c79210a817aeb54f7a526.tar.gz",  # 23.x(2023-06-13)
        ],
        patches = [
            # When using a toolchain on Windows where the runtime libraries are
            # not installed system-wide, it is necessary to specify
            # `use_default_shell_env = True` in order to be able to execute
            # compiled binaries.
            "//third_party:com_google_protobuf_upb/patches/use_default_shell_env.diff",

            # bootstrap.diff includes the pre-compiled source for plugin.proto and
            # descriptor.proto, used for bootstrapping the cmake build, which also breaks
            # a big dependency on @com_google_protobuf.
            "//third_party:com_google_protobuf_upb/patches/bootstrap.diff",
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@utf8_range": "@com_google_protobuf_utf8_range",
        },
        # CMake support in upb is experimental; however we only need upb support for gRPC.
        cmake_name = "upb",
        cmake_extra_build_file = Label("//third_party:com_google_protobuf_upb/cmake_extra.BUILD.bazel"),
        cmake_enable_system_package = False,
        cmake_target_mapping = {
            "//:reflection": "upb::reflection",
            "//:textformat": "upb::textformat",
            "//:collections": "upb::collections",
            "//:upb": "upb::upb",
        },
        bazel_to_cmake = {
            "args": [
                "--ignore-library=//bazel:amalgamation.bzl",
                "--ignore-library=//bazel:py_proto_library.bzl",
                "--ignore-library=//bazel:python_downloads.bzl",
                "--ignore-library=//bazel:system_python.bzl",
                "--ignore-library=//bazel:workspace_deps.bzl",
                "--ignore-library=//lua:lua_proto_library.bzl",
                "--ignore-library=//protos/bazel:upb_cc_proto_library.bzl",
                "--ignore-library=//python/dist:dist.bzl",
                "--ignore-library=//python:py_extension.bzl",
                "--ignore-library=//benchmarks:build_defs.bzl",
                "--target=//:json",
                "--target=//:reflection",
                "--target=//:textformat",
                "--target=//:message",
                "--target=//:upb",
                "--target=//:port",
                "--target=//:collections",
                "--target=//upbc:protoc-gen-upb",
                "--target=//upbc:protoc-gen-upbdefs",
                # support libraries
                "--target=//:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
                "--target=//:generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
                "--target=//:upb_proto_library_copts__for_generated_code_only_do_not_use",
                # bootstrap
                "--target=//:cmake_descriptor_upb",
                "--target=//:cmake_descriptor_upbdefs",
                "--target=//:cmake_plugin_upb",
                "--target=//:cmake_plugin_upbdefs",
                "--bind=@com_google_protobuf//src/google/protobuf/compiler:code_generator=@com_google_protobuf//:protoc_lib",
            ],
            "exclude": ["lua/**", "protos/**", "python/**", "benchmarks/**", "cmake/**"],
        },
    )

# NOTE: --bind allows remapping of dependencies to work with the system dependencies and the CMake
# function find_package(Protobuf).  When updating upb, review the dependencies to see if any need
# to be mapped to their equivalents.
#
# ./bazelisk.py query -k "allpaths(@com_google_protobuf_upb//..., @com_google_protobuf//...)"
#  --notool_deps --output=graph  |
#   egrep '["]@com_google_protobuf_upb//[^"]*["]\s*->\s*["]@com_google_protobuf//[^"]*["]'
