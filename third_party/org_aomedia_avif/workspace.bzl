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

load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//:cmake_helpers.bzl", "cmake_add_dep_mapping", "cmake_fetch_content_package")

def repo():
    maybe(
        third_party_http_archive,
        name = "org_aomedia_avif",
        urls = [
            "https://github.com/AOMediaCodec/libavif/archive/v0.10.1.tar.gz",
        ],
        sha256 = "66e82854ceb84a3e542bc140a343bc90e56c68f3ecb4fff63e636c136ed9a05e",
        strip_prefix = "libavif-0.10.1",
        build_file = Label("//third_party:org_aomedia_avif/libavif.BUILD.bazel"),
        system_build_file = Label("//third_party:org_aomedia_avif/system.BUILD.bazel"),
    )

cmake_fetch_content_package(
    name = "avif",
    settings = [
        ("AVIF_BUILD_AOM", "ON"),
        ("AVIF_CODEC_DAV1D", "OFF"),  # TODO: Change to ON
        ("AVIF_CODEC_AOM_DECODE", "ON"),
        ("AVIF_CODEC_AOM_ENCODE", "ON"),
        ("AVIF_LIBYUV_ENABLED", "ON"),
    ],
)

cmake_add_dep_mapping(target_mapping = {
    "@org_aomedia_avif//:avif": "avif",
})
