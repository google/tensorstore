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

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_libyuv",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/chromium.googlesource.com/libyuv/libyuv/+archive/97bd3bba83e3eb4df85e8e243ceb1abebc190a43.tar.gz",
        ],
        # googlesource does not cache archive files; the sha256 is only valid for the mirror.
        sha256 = "668d771967903caa559d749bc188e91bab4d355b3631b013068a582ab78b0368",
        build_file = Label("//third_party:com_google_libyuv/libyuv.BUILD.bazel"),
        cmake_name = "libyuv",
        cmake_target_mapping = {
            ":libyuv": "LIBYUV::LIBYUV",
        },
        bazel_to_cmake = {},
        cmake_package_redirect_libraries = {
            "LIBYUV": "LIBYUV::LIBYUV",
        },
    )
