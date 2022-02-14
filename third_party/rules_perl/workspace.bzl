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

# REPO = https://github.com/bazelbuild/rules_perl/refs/head/main

def repo():
    maybe(
        third_party_http_archive,
        name = "rules_perl",
        urls = [
            "https://github.com/bazelbuild/rules_perl/archive/e288d228930c83081a697076f7fa8e7f08b52a3a.tar.gz",
        ],
        sha256 = "ff85afdf3e6f1cb49fdfba89c8954935598f31380134d57341ae8bcba978260a",
        strip_prefix = "rules_perl-e288d228930c83081a697076f7fa8e7f08b52a3a",
        patches = [
            "//third_party:rules_perl/patches/pull_request_38.diff",
        ],
        patch_args = ["-p1"],
    )
