# Copyright 2023 The TensorStore Authors
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
        name = "com_google_googleapis",
        strip_prefix = "googleapis-61220850831d674c6a025e5e315f18f0922a352e",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/googleapis/googleapis/archive/61220850831d674c6a025e5e315f18f0922a352e.tar.gz",  # master(2023-02-28)
        ],
        sha256 = "870ae54700826bb9d7ef4a5c6696dc0feb032f7a50842961ed77ff5034d0d2b2",
        repo_mapping = {
            "@com_google_googleapis_imports": "@local_proto_mirror",
        },
        cmake_name = "Googleapis",
        bazel_to_cmake = {
            "args": [
                "--exclude-target=//:build_gen",
                "--target=//google/api:all",
                "--target=//google/api/expr/v1alpha1:all",
                "--target=//google/rpc:all",
                "--target=//google/storage/v2:all",
            ],
            "exclude": GOOGLEAPIS_EXCLUDES,
        },
    )

# Above, we set --target to the storage targets, then add excludes for all paths which
# storage does not depend on by checking with a bazel dependency query:
# ./bazelisk.py  query -k 'deps(tensorstore/kvstore/gcs:gcs_testbench)' --output=label -k | grep googleapis
#
GOOGLEAPIS_EXCLUDES = [
    "google/firebase/**",
    "google/devtools/**",
    "google/apps/**",
    "google/dataflow/**",
    "google/geo/**",
    "google/logging/**",
    "google/genomics/**",
    "google/analytics/**",
    "google/maps/**",
    "google/area120/**",
    "google/storagetransfer/**",
    "google/watcher/**",
    "google/example/**",
    "google/cloud/**",
    "google/monitoring/**",
    "google/container/**",
    "google/firestore/**",
    "google/privacy/**",
    "google/actions/**",
    "google/streetview/**",
    "google/identity/**",
    "google/pubsub/**",
    "google/assistant/**",
    "google/datastore/**",
    "google/appengine/**",
    "google/bytestream/**",
    "google/chromeos/**",
    "google/longrunning/**",
    "google/partner/**",
    "google/networking/**",
    "google/search/**",
    "google/ads/**",
    "google/spanner/**",
    "google/home/**",
    "google/bigtable/**",
    "google/chat/**",
    "grafeas/**",
]
