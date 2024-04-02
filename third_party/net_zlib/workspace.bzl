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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load(
    "//third_party:repo.bzl",
    "third_party_http_archive",
)

# Using chromium-zlib, source from: https://chromium.googlesource.com/chromium/src/third_party/zlib/

def repo():
    maybe(
        third_party_http_archive,
        name = "net_zlib",
        # We use the chromium fork of zlib, but generally it tracks here:
        doc_name = "chromium-zlib",
        doc_version = "3787595bbbd3a374613713164db935e8331f5825",
        doc_homepage = "https://chromium.googlesource.com/chromium/src/third_party/zlib/",
        # googlesource does not cache archive files; the sha256 is only valid for the mirror.
        sha256 = "096325e0c605fcb23aa578e9d259f5e577da1f1d0bdd68e9887974072c120a09",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/chromium.googlesource.com/chromium/src/third_party/zlib/+archive/3787595bbbd3a374613713164db935e8331f5825.tar.gz",
        ],
        build_file = Label("//third_party:net_zlib/zlib.BUILD.bazel"),
        system_build_file = Label("//third_party:net_zlib/system.BUILD.bazel"),
        cmake_name = "ZLIB",
        bazel_to_cmake = {},
        cmake_target_mapping = {
            "@net_zlib//:zlib": "ZLIB::ZLIB",
        },
        cmake_package_redirect_libraries = {
            "ZLIB": "ZLIB::ZLIB",
        },
    )
