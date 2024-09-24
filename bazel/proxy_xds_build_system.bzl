# Copyright 2024 The TensorStore Authors
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

"""
Replacement for @com_github_cncf_udpa//bazel:api_build_system.bzl used by bazel_to_cmake
to simplify the CMake target generation.

This does not provide all the functionality used by the cncf/xds repository.
"""

load(
    "@tensorstore//bazel:tensorstore.bzl",
    _tensorstore_cc_proto_library = "tensorstore_cc_proto_library",
    _tensorstore_proto_library = "tensorstore_proto_library",
)

_GAPI = "@com_google_googleapis//google/api"

# This maps from the Bazel proto_library target to the C++ language binding target for external dependencies.
EXTERNAL_PROTO_CC_BAZEL_DEP_MAP = {
    _GAPI + "expr/v1alpha1:checked_proto": _GAPI + "expr/v1alpha1:checked_cc_proto",
    _GAPI + "expr/v1alpha1:syntax_proto": _GAPI + "expr/v1alpha1:syntax_cc_proto",
    "@cel-spec//proto/cel/expr:checked_proto": "@cel-spec//proto/cel/expr:checked_cc_proto",
    "@cel-spec//proto/cel/expr:syntax_proto": "@cel-spec//proto/cel/expr:syntax_cc_proto",
}

_CC_PROTO_SUFFIX = "_cc_proto"

_COMMON_PROTO_DEPS = [
    "@com_google_protobuf//:any_proto",
    "@com_google_protobuf//:descriptor_proto",
    "@com_google_protobuf//:duration_proto",
    "@com_google_protobuf//:empty_proto",
    "@com_google_protobuf//:struct_proto",
    "@com_google_protobuf//:timestamp_proto",
    "@com_google_protobuf//:wrappers_proto",
    "@com_google_googleapis//google/api:http_proto",
    "@com_google_googleapis//google/rpc:status_proto",
    "@com_envoyproxy_protoc_gen_validate//validate:validate_proto",
]

def _proto_mapping(dep, proto_dep_map, proto_suffix):
    mapped = proto_dep_map.get(dep)
    if mapped == None:
        prefix = "@" + Label(dep).workspace_name if not dep.startswith("//") else ""
        return prefix + "//" + Label(dep).package + ":" + Label(dep).name + proto_suffix
    return mapped

def _cc_proto_mapping(dep):
    return _proto_mapping(dep, EXTERNAL_PROTO_CC_BAZEL_DEP_MAP, _CC_PROTO_SUFFIX)

def xds_proto_package(
        name = "pkg",
        srcs = [],
        deps = [],
        has_services = False,
        visibility = ["//visibility:public"]):
    name = "pkg"
    if srcs == []:
        srcs = native.glob(["*.proto"])

    _tensorstore_proto_library(
        name = name,
        srcs = srcs,
        deps = deps + _COMMON_PROTO_DEPS,
        visibility = visibility,
    )

    _tensorstore_cc_proto_library(
        name = name + _CC_PROTO_SUFFIX,
        cc_deps = [_cc_proto_mapping(dep) for dep in deps] + [
            "@com_google_googleapis//google/api:http_cc_proto",
            "@com_google_googleapis//google/api:httpbody_cc_proto",
            "@com_google_googleapis//google/rpc:status_cc_proto",
        ],
        deps = [name],
        visibility = ["//visibility:public"],
    )

def xds_cc_test(**kwargs):
    pass

def xds_go_test(**kwargs):
    pass
