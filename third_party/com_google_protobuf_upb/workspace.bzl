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
        sha256 = "ac79e4540f04a6de945cd827b2e1be71f7d232c46f7d09621de2da1003c763d9",
        strip_prefix = "upb-f3a0cc49da29dbdbd09b3325c2834139540f00fa",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/protocolbuffers/upb/archive/f3a0cc49da29dbdbd09b3325c2834139540f00fa.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/f3a0cc49da29dbdbd09b3325c2834139540f00fa.tar.gz",  # main(2022-11-18)
        ],
        patches = [
            "//third_party:com_google_protobuf_upb/patches/build.diff",
            # When using a toolchain on Windows where the runtime libraries are
            # not installed system-wide, it is necessary to specify
            # `use_default_shell_env = True` in order to be able to execute
            # compiled binaries.
            "//third_party:com_google_protobuf_upb/patches/use_default_shell_env.diff",
            # Fixes https://github.com/google/tensorstore/issues/86
            #
            # Without this patch, upb copies an uninitialized value, but the
            # copy is never actually used.  That is technically undefined
            # behavior.  In normal builds it is benign but in certain MSVC
            # sanitizer builds it leads to an error.
            "//third_party:com_google_protobuf_upb/patches/fix-uninitialized-value-copy.diff",
        ],
        patch_args = ["-p1"],
        cmake_name = "upb",
        cmake_enable_system_package = False,
        cmake_target_mapping = {
            "//:reflection": "upb::reflection",
            "//:textformat": "upb::textformat",
            "//:upb": "upb::upb",
            "//upbc:protoc-gen-upb": "upb::protoc_gen_upb",
            "//upbc:protoc-gen-upbdefs": "upb::protoc_gen_upbdefs",
        },
        bazel_to_cmake = {
            "args": [
                "--ignore-library=//bazel:amalgamation.bzl",
                "--ignore-library=//bazel:py_proto_library.bzl",
                "--ignore-library=//lua:lua_proto_library.bzl",
                "--ignore-library=//protos/bazel:upb_cc_proto_library.bzl",
                "--ignore-library=//python/dist:dist.bzl",
                "--ignore-library=//python:py_extension.bzl",
                "--ignore-library=//benchmarks:build_defs.bzl",
                # bootstrap
                "--bind=@com_google_protobuf//src/google/protobuf/compiler:code_generator=@com_google_protobuf//:protoc_lib",
                "--bind=@com_google_protobuf//:descriptor_proto=@com_google_protobuf_upb//bootstrap/google/protobuf:descriptor_proto",
                "--bind=@com_google_protobuf//:descriptor_proto_srcs=@com_google_protobuf_upb//bootstrap/google/protobuf:descriptor_proto_src",
                #"--bind=@com_google_protobuf_upb//:descriptor_upb_proto=@com_google_protobuf_upb//bootstrap/google/protobuf:descriptor_upb_proto",
                "--bind=@com_google_protobuf//:compiler_plugin_proto=@com_google_protobuf_upb//bootstrap/google/protobuf/compiler:plugin_proto",
                "--bind=@com_google_protobuf//src/google/protobuf/compiler:plugin_proto_src=@com_google_protobuf_upb//bootstrap/google/protobuf/compiler:plugin_proto_src",
                #"--bind=@com_google_protobuf_upb//upbc:plugin_upb_proto=@com_google_protobuf_upb//bootstrap/google/protobuf/compiler:plugin_upb_proto",
                "--target=//bootstrap/google/protobuf:all",
                "--target=//bootstrap/google/protobuf/compiler:all",
                # bootstrap
                "--target=//:json",
                "--target=//:reflection",
                "--target=//:textformat",
                "--target=//:message",
                "--target=//:upb",
                "--target=//:port",
                "--target=//:collections",
                "--target=//upbc:protoc-gen-upb_stage0",
                "--target=//upbc:protoc-gen-upb_stage1",
                "--target=//upbc:protoc-gen-upb",
                "--target=//upbc:protoc-gen-upbdefs",
                "--target=//:descriptor_upb_proto",
                "--target=//:descriptor_upb_proto_reflection",
                # support libraries
                "--target=//:generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
                "--target=//:generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me",
                "--target=//:upb_proto_library_copts__for_generated_code_only_do_not_use",
            ],
            "exclude": ["lua/**", "protos/**", "python/**", "rust/**", "benchmarks/**", "cmake/**"],
        },
    )
