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
        name = "com_google_boringssl",
        urls = [
            "https://github.com/google/boringssl/archive/bdbe37905216bea8dd4d0fdee93f6ee415d3aa15.tar.gz",  # master-with-bazel(2021-01-09)
        ],
        sha256 = "ce183cb587c0a0f5982e441dff91cb5456d4c85cfa3fb12816e7a93f20645e51",
        strip_prefix = "boringssl-bdbe37905216bea8dd4d0fdee93f6ee415d3aa15",
        system_build_file = Label("//third_party/com_google_boringssl:system.BUILD.bazel"),
    )
