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
load("//:cmake_helpers.bzl", "cmake_fetch_content_package")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_libyuv",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/chromium.googlesource.com/libyuv/libyuv/+archive/2525698acba9bf9b701ba6b4d9584291a1f62257.tar.gz",
            # "https://chromium.googlesource.com/libyuv/libyuv/+archive/2525698acba9bf9b701ba6b4d9584291a1f62257.tar.gz",
        ],
        # googlesource does not cache archive files; the sha256 is only valid for the mirror.
        sha256 = "43dfa4511332424dfad1ce48be645bb063faf0e48f0b3e6c07c2cb308db80f9e",
        build_file = Label("//third_party:com_google_libyuv/libyuv.BUILD.bazel"),
    )

# yuv is only used by avif

cmake_fetch_content_package(name = "libyuv")
