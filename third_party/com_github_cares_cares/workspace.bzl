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

# Must match grpc c-ares, see grpc/bazel/grpc_deps.bzl

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_cares_cares",
        sha256 = "ec76c5e79db59762776bece58b69507d095856c37b81fd35bfb0958e74b61d93",
        strip_prefix = "c-ares-6654436a307a5a686b008c1d4c93b0085da6e6d8",
        urls = [
            "https://storage.googleapis.com/grpc-bazel-mirror/github.com/c-ares/c-ares/archive/6654436a307a5a686b008c1d4c93b0085da6e6d8.tar.gz",
            "https://github.com/c-ares/c-ares/archive/6654436a307a5a686b008c1d4c93b0085da6e6d8.tar.gz",
        ],
        build_file = Label("//third_party:com_github_cares_cares/cares.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_cares_cares/system.BUILD.bazel"),
        cmake_name = "c-ares",
        cmake_target_mapping = {
            "//:ares": "c-ares::cares",
        },
        bazel_to_cmake = {
            "aliased_targets_only": True,
        },
    )
