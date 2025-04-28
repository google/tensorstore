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
        name = "blake3",
        strip_prefix = "BLAKE3-1.8.2",
        sha256 = "6b51aefe515969785da02e87befafc7fdc7a065cd3458cf1141f29267749e81f",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/BLAKE3-team/BLAKE3/archive/1.8.2.tar.gz",
        ],
        build_file = Label("//third_party:blake3/blake3.BUILD.bazel"),
        system_build_file = Label("//third_party:blake3/system.BUILD.bazel"),
        cmake_name = "BLAKE3",
        cmake_languages = ["ASM", "ASM_MASM"],
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@blake3//:blake3": "BLAKE3::BLAKE3",
        },
    )
