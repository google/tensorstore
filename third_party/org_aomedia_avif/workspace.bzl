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
            "https://github.com/AOMediaCodec/libavif/archive/d9cffc5f46b62aeff46eebf51449726386d6c485.tar.gz",
        ],
        sha256 = "6dce70530fa750544ba842e40555825e8894e8f805cf5c458ee0642a60d160e5",
        strip_prefix = "libavif-d9cffc5f46b62aeff46eebf51449726386d6c485",
        build_file = Label("//third_party:org_aomedia_avif/libavif.BUILD.bazel"),
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
