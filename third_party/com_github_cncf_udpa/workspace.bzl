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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_cncf_udpa",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/cncf/xds/archive/e9ce68804cb4e64cab5a52e3c8baf840d4ff87b7.tar.gz",  # main(2023-06-23)
        ],
        sha256 = "0d33b83f8c6368954e72e7785539f0d272a8aba2f6e2e336ed15fd1514bc9899",
        strip_prefix = "xds-e9ce68804cb4e64cab5a52e3c8baf840d4ff87b7",
        repo_mapping = {
            "@io_bazel_rules_go": "@local_proto_mirror",
            "@com_envoyproxy_protoc_gen_validate": "@local_proto_mirror",
        },
        # CMake options
        cmake_name = "udpa",
        cmake_extra_build_file = Label("//third_party:com_github_cncf_udpa/cmake_extra.BUILD.bazel"),
        bazel_to_cmake = {
            "args": ["--target=" + p + ":all" for p in _PACKAGES],
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
