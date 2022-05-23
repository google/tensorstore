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

# source = https://aomedia.googlesource.com/aom/

def repo():
    maybe(
        third_party_http_archive,
        name = "org_aomedia_aom",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/aomedia.googlesource.com/aom/+archive/287164de79516c25c8c84fd544f67752c170082a.tar.gz",
            # "https://aomedia.googlesource.com/aom/+archive/287164de79516c25c8c84fd544f67752c170082a.tar.gz",
        ],
        # googlesource does not cache archive files; the sha256 is only valid for the mirror.
        sha256 = "7508dcde9e260621862639fb6a2d3154bcbd10e65d43f107595c6a6aaed55455",
        build_file = Label("//third_party:org_aomedia_aom/libaom.BUILD.bazel"),
    )

# Used by avif
cmake_fetch_content_package(
    name = "org_aomedia_aom",
    configure_command = "",
    build_command = "",
    make_available = False,
)
