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
        sha256 = "ac16b9ac09338c0b2971c2e97583c6c61cd9e0addb945cf037531aefcf568fbc",
        strip_prefix = "riegeli-36e220cd77bf83e3399f60e78d561e3820ba025d",
        urls = [
            "https://github.com/google/riegeli/archive/36e220cd77bf83e3399f60e78d561e3820ba025d.tar.gz",  # master(2021-10-06)
        ],
    )
