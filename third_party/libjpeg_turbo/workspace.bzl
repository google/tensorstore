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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

def repo():
    maybe(
        third_party_http_archive,
        # Note: The generic name "jpeg" is used in place of the more canonical
        # "org_libjpeg_turbo" because this repository may actually refer to the
        # system jpeg.
        name = "libjpeg_turbo",
        strip_prefix = "libjpeg-turbo-2.1.5.1",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/libjpeg-turbo/libjpeg-turbo/archive/2.1.5.1.tar.gz",
        ],
        sha256 = "61846251941e5791005fb7face196eec24541fce04f12570c308557529e92c75",
        doc_version = "2.1.5.1",
        build_file = Label("//third_party:libjpeg_turbo/jpeg.BUILD.bazel"),
        system_build_file = Label("//third_party:libjpeg_turbo/system.BUILD.bazel"),
        cmake_name = "JPEG",
        # libjpeg-turbo maintainers do not wish to support subproject builds
        # through CMake, so use bazel_to_cmake instead:
        # https://github.com/libjpeg-turbo/libjpeg-turbo/pull/622
        bazel_to_cmake = {},
        cmake_languages = ["ASM_NASM"],
        cmake_target_mapping = {
            "//:jpeg": "JPEG::JPEG",
        },
    )
