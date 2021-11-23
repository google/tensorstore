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

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_riegeli",
        strip_prefix = "riegeli-b15854f6cac8e2d6076ca11b7533765f85e27d98",
        urls = [
            "https://github.com/google/riegeli/archive/b15854f6cac8e2d6076ca11b7533765f85e27d98.tar.gz",  # master(2021-11-05)
        ],
        sha256 = "b91c67efb60905d9d3f37951c288da4556710cc84ec61f9697ebdc700635726d",
    )
