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

# Should be compatible with grpc/bazel/grpc_deps.bzl c-ares.

def repo():
    maybe(
        third_party_http_archive,
        name = "c-ares",
        sha256 = "dcd919635f01b7c8c9c2f5fb38063cd86500f7c6d4d32ecf4deff5e3497fb157",
        strip_prefix = "c-ares-1.34.5",
        urls = mirror_url("https://github.com/c-ares/c-ares/archive/v1.34.5.tar.gz"),
        build_file = Label("//third_party:c_ares/cares.BUILD.bazel"),
        system_build_file = Label("//third_party:c_ares/system.BUILD.bazel"),
        cmake_name = "c-ares",
        cmake_target_mapping = {
            "//:ares": "c-ares::cares",
        },
        bazel_to_cmake = {
            "aliased_targets_only": True,
        },
    )
