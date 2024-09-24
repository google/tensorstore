# Copyright 2023 The TensorStore Authors
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
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_cncf_udpa",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/cncf/xds/archive/b4127c9b8d78b77423fd25169f05b7476b6ea932.tar.gz",  # main(2024-09-11)
        ],
        sha256 = "aa5f1596bbef3f277dcf4700e4c1097b34301ae66f3b79cd731e3adfbaff2f8f",
        strip_prefix = "xds-b4127c9b8d78b77423fd25169f05b7476b6ea932",
        repo_mapping = {
            "@io_bazel_rules_go": "@local_proto_mirror",
            "@com_envoyproxy_protoc_gen_validate": "@local_proto_mirror",
            "@dev_cel": "@cel-spec",
        },
        # CMake options
        cmake_name = "udpa",
        cmake_extra_build_file = Label("//third_party:com_github_cncf_udpa/cmake_extra.BUILD.bazel"),
        bazel_to_cmake = {
            "args": [
                "--bind=@com_github_cncf_udpa//bazel:api_build_system.bzl=@tensorstore//bazel:proxy_xds_build_system.bzl",
            ] + [
                "--target=" + p + ":pkg"
                for p in _PACKAGES
            ] + [
                "--target=" + p + ":pkg_cc_proto"
                for p in _PACKAGES
            ] + [
                "--target=" + p + ":pkg__upb_library"
                for p in _PACKAGES
            ] + [
                "--target=" + p + ":pkg__upbdefs_library"
                for p in _PACKAGES
            ],
        },
    )

_PACKAGES = [
    "//udpa/annotations",
    "//xds/data/orca/v3",
    "//xds/service/orca/v3",
    "//xds/type/v3",
    "//xds/type/matcher/v3",
    "//xds/core/v3",
]
