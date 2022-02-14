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

def repo():
    maybe(
        third_party_http_archive,
        name = "org_aomedia_avif",
        urls = [
            "https://github.com/AOMediaCodec/libavif/archive/d9cffc5f46b62aeff46eebf51449726386d6c485.tar.gz",
        ],
        strip_prefix = "libavif-d9cffc5f46b62aeff46eebf51449726386d6c485",
        build_file = Label("//third_party:org_aomedia_avif/libavif.BUILD.bazel"),
    )
