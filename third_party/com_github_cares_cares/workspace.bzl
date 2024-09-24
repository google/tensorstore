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
load("//third_party:repo.bzl", "third_party_http_archive")

# Should be compatible with grpc/bazel/grpc_deps.bzl c-ares.

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_cares_cares",
        sha256 = "b3d127d8357863eb465053ce9308b79d9b00314f92ec09df056221a1a45c2fef",
        strip_prefix = "c-ares-1.33.1",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/c-ares/c-ares/archive/v1.33.1.tar.gz",
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
