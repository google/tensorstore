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
        name = "net_sourceforge_half",
        urls = [
            "http://sourceforge.net/projects/half/files/half/2.1.0/half-2.1.0.zip",
        ],
        sha256 = "ad1788afe0300fa2b02b0d1df128d857f021f92ccf7c8bddd07812685fa07a25",
        build_file = Label("//third_party:net_sourceforge_half/bundled.BUILD.bazel"),
    )
