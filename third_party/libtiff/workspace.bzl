# Copyright 2022 The TensorStore Authors
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
    "mirror_url",
    "third_party_http_archive",
)

# Homepages http://www.simplesystems.org/libtiff/
# TODO: Use mirror the gitlab bundle: "https://gitlab.com/libtiff/libtiff"
def repo():
    maybe(
        third_party_http_archive,
        name = "libtiff",
        urls = mirror_url("https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz"),
        sha256 = "f698d94f3103da8ca7438d84e0344e453fe0ba3b7486e04c5bf7a9a3fabe9b69",
        doc_version = "4.7.0",
        strip_prefix = "tiff-4.7.1",
        build_file = Label("//third_party:libtiff/libtiff.BUILD.bazel"),
        system_build_file = Label("//third_party:libtiff/system.BUILD.bazel"),
        remove_paths = ["VERSION"],
        cmake_name = "TIFF",
        cmake_target_mapping = {
            "//:tiff": "TIFF::TIFF",
        },
        bazel_to_cmake = {},
    )
