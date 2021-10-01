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
        name = "com_google_riegeli",
        sha256 = "f145eaa36f4c4d4ca70d2304cd8a301c7acf0dab5ee5e78011591a1528ff046d",
        strip_prefix = "riegeli-b8a9bf249ceb6d8267161c3de0aad4b99fc605ee",
        urls = [
            "https://github.com/google/riegeli/archive/b8a9bf249ceb6d8267161c3de0aad4b99fc605ee.tar.gz",  # master(2021-09-13)
        ],
    )
