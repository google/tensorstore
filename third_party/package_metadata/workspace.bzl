# Copyright 2026 The TensorStore Authors
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

def repo():
    maybe(
        third_party_http_archive,
        name = "package_metadata",
        sha256 = "0e89367f1cb6d93a5a1afea4b55b11ea6b28f63f653b47154153677ca7d4afea",
        strip_prefix = "supply-chain-0.0.3/metadata",
        urls = mirror_url("https://github.com/bazel-contrib/supply-chain/releases/download/v0.0.3/supply-chain-v0.0.3.tar.gz"),
    )
