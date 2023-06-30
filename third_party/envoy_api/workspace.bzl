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

load("//third_party:repo.bzl", "third_party_http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "envoy_api",
        urls = [
            "https://storage.googleapis.com/grpc-bazel-mirror/github.com/envoyproxy/data-plane-api/archive/68d4315167352ffac71f149a43b8088397d3f33d.tar.gz",
            "https://github.com/envoyproxy/data-plane-api/archive/68d4315167352ffac71f149a43b8088397d3f33d.tar.gz",
        ],
        sha256 = "3c7372b5cb33e5e5cc3afd82573fc6275f9a2cac8b1530e1af14f52f34047328",
        strip_prefix = "data-plane-api-68d4315167352ffac71f149a43b8088397d3f33d",
        repo_mapping = {
            "@com_envoyproxy_protoc_gen_validate": "@local_proto_mirror",
            "@io_bazel_rules_go": "@local_proto_mirror",
            "@opencensus_proto": "@local_proto_mirror",
        },

        # CMake options
        cmake_name = "envoy",
        cmake_extra_build_file = Label("//third_party:envoy_api/cmake_extra.BUILD.bazel"),
        bazel_to_cmake = {
            "args": ["--target=" + p + ":all" for p in _PACKAGES],
        },
    )

_PACKAGES = [
    "//envoy/admin/v3",
    "//envoy/config/cluster/v3",
    "//envoy/config/core/v3",
    "//envoy/config/endpoint/v3",
    "//envoy/config/listener/v3",
    "//envoy/config/rbac/v3",
    "//envoy/config/route/v3",
    "//envoy/extensions/clusters/aggregate/v3",
    "//envoy/extensions/filters/common/fault/v3",
    "//envoy/extensions/filters/http/fault/v3",
    "//envoy/extensions/filters/http/rbac/v3",
    "//envoy/extensions/filters/http/router/v3",
    "//envoy/extensions/filters/http/stateful_session/v3",
    "//envoy/extensions/filters/network/http_connection_manager/v3",
    "//envoy/extensions/http/stateful_session/cookie/v3",
    "//envoy/extensions/load_balancing_policies/common/v3",
    "//envoy/extensions/load_balancing_policies/client_side_weighted_round_robin/v3",
    "//envoy/extensions/load_balancing_policies/ring_hash/v3",
    "//envoy/extensions/load_balancing_policies/wrr_locality/v3",
    "//envoy/extensions/transport_sockets/tls/v3",
    "//envoy/service/discovery/v3",
    "//envoy/service/load_stats/v3",
    "//envoy/service/status/v3",
    "//envoy/type/http/v3",
    "//envoy/type/matcher/v3",
    "//envoy/type/v3",
]
