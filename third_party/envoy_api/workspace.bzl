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

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "envoy_api",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/envoyproxy/data-plane-api/archive/d9c5e84658eef279e9a021ff0517f8f8ee35d79a.tar.gz",  # main(2025-04-26)
            "https://github.com/envoyproxy/data-plane-api/archive/d9c5e84658eef279e9a021ff0517f8f8ee35d79a.tar.gz",
        ],
        sha256 = "a9f1eb76c8e8153ec81ce04403c8829b42786060b7bb1dcc91b597220b71eaf6",
        strip_prefix = "data-plane-api-d9c5e84658eef279e9a021ff0517f8f8ee35d79a",
        repo_mapping = {
            "@com_envoyproxy_protoc_gen_validate": "@local_proto_mirror",
            "@io_bazel_rules_go": "@local_proto_mirror",
            "@opencensus_proto": "@local_proto_mirror",
            "@com_github_cncf_xds": "@xds",
            "@com_google_googleapis": "@googleapis",
            "@com_github_grpc_grpc": "@grpc",
            "@rules_python": "@local_proto_mirror",
        },
        # CMake options
        cmake_name = "envoy",
        cmake_extra_build_file = Label("//third_party:envoy_api/cmake_extra.BUILD.bazel"),
        bazel_to_cmake = {
            "args": [
                "--ignore-library=//bazel/cc_proto_descriptor_library:builddefs.bzl",
                "--ignore-library=@io_bazel_rules_go//go:def.bzl",
                "--ignore-library=@io_bazel_rules_go//proto:def.bzl",
                "--ignore-library=@grpc//bazel:python_rules.bzl",
            ] + ["--target=" + p + ":all" for p in _PACKAGES],
        },
    )

# This list mostly comes from grpc dependencies:
# $ find grpc/ -type f -name BUILD | xargs grep "@envoy |
#     awk -F: '{print $2}' | sort | uniq
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
    "//envoy/extensions/filters/http/gcp_authn/v3",
    "//envoy/extensions/filters/http/rbac/v3",
    "//envoy/extensions/filters/http/router/v3",
    "//envoy/extensions/filters/http/stateful_session/v3",
    "//envoy/extensions/filters/network/http_connection_manager/v3",
    "//envoy/extensions/http/stateful_session/cookie/v3",
    "//envoy/extensions/load_balancing_policies/client_side_weighted_round_robin/v3",
    "//envoy/extensions/load_balancing_policies/cluster_provided/v3",
    "//envoy/extensions/load_balancing_policies/common/v3",
    "//envoy/extensions/load_balancing_policies/least_request/v3",
    "//envoy/extensions/load_balancing_policies/maglev/v3",
    "//envoy/extensions/load_balancing_policies/pick_first/v3",
    "//envoy/extensions/load_balancing_policies/random/v3",
    "//envoy/extensions/load_balancing_policies/ring_hash/v3",
    "//envoy/extensions/load_balancing_policies/round_robin/v3",
    "//envoy/extensions/load_balancing_policies/subset/v3",
    "//envoy/extensions/load_balancing_policies/wrr_locality/v3",
    "//envoy/extensions/rbac/audit_loggers/stream/v3",
    "//envoy/extensions/transport_sockets/http_11_proxy/v3",
    "//envoy/extensions/transport_sockets/tls/v3",
    "//envoy/extensions/upstreams/http/v3",
    "//envoy/extensions/upstreams/tcp/v3",
    "//envoy/service/discovery/v3",
    "//envoy/service/load_stats/v3",
    "//envoy/service/status/v3",
    "//envoy/type/http/v3",
    "//envoy/type/matcher/v3",
    "//envoy/type/v3",
]
