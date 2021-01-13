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
        name = "org_blosc_cblosc",
        strip_prefix = "c-blosc-1.21.0",
        urls = [
            "https://github.com/Blosc/c-blosc/archive/v1.21.0.zip",
        ],
        sha256 = "09dee9dc4d1b85463b87d92f1494a44d23eafadbda2580b76e327eaa59f8a9ef",
        build_file = Label("//third_party:org_blosc_cblosc/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:org_blosc_cblosc/system.BUILD.bazel"),
    )
