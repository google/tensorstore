# Copyright 2020 The TensorStore Authors
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

# Examples of building libavif using bazel are hard to find.
# https://github.com/tensorflow/io

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        name = "org_aomedia_avif",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/AOMediaCodec/libavif/archive/d1c26facaf5a8a97919ceee06814d05d10e25622.tar.gz",  # 1.0.1(2023-08-30)
        ],
        sha256 = "ace07eede0af2960dbad3b46f66a2902cc622ccdf4c404ad92620e38050498ba",
        strip_prefix = "libavif-d1c26facaf5a8a97919ceee06814d05d10e25622",
        build_file = Label("//third_party:org_aomedia_avif/libavif.BUILD.bazel"),
        system_build_file = Label("//third_party:org_aomedia_avif/system.BUILD.bazel"),
        cmake_name = "AVIF",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "//:avif": "AVIF::AVIF",
        },
        cmake_package_redirect_libraries = {
            "AVIF": "AVIF::AVIF",
        },
    )
