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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        # Note: The generic name "jpeg" is used in place of the more canonical
        # "org_libjpeg_turbo" because this repository may actually refer to the
        # system jpeg.
        name = "jpeg",
        strip_prefix = "libjpeg-turbo-2.1.4",
        urls = [
            "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.1.4.tar.gz",
        ],
        sha256 = "a78b05c0d8427a90eb5b4eb08af25309770c8379592bb0b8a863373128e6143f",
        build_file = Label("//third_party:jpeg/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:jpeg/system.BUILD.bazel"),
        cmake_name = "JPEG",
        # libjpeg-turbo maintainers do not wish to support subproject builds
        # through CMake, so use bazel_to_cmake instead:
        # https://github.com/libjpeg-turbo/libjpeg-turbo/pull/622
        bazel_to_cmake = {},
        cmake_languages = ["ASM_NASM"],
        cmake_target_mapping = {
            "@jpeg//:jpeg": "JPEG::JPEG",
        },
    )
