# Copyright 2022 The TensorStore Authors
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

"""Variation of @com_github_grpc_grpc//:bazel/cc_grpc_library.bzl.

This version only generates the gRPC target and the grpc.cc and grpc.h
files, and relies on a separate proto_library() target (as when grpc_only=True).
It also adds support for service_namesspace in the grpc targets.

TODO: Try and upstream changes to make this file unnecessary.
"""

load("@com_github_grpc_grpc//bazel:generate_cc.bzl", "generate_cc")

def cc_grpc_library(name, srcs, deps, visibility = None, service_namespace = None, **kwargs):
    """Generates C++ grpc classes from a .proto file.

    Note this rule only generates gRPC interfaces. You need to have proto_library
    To generate the classes for the proto messages. It is an error to use this
    with a proto_library with cc_grpc_version.

    Args:
      name: (str) name of rule
      srcs: (list) a single proto_library, which wraps the .proto file and has
            cc_api_version = 2
      deps: (list) typically a single proto_library or a cc_proto_library for the
            proto_library in srcs. The list will be added to the cc_library's deps
            list.
      visibility: (list) visibility of the library.
      service_namespace: (str) additional namespace to avoid collision with stubby
            services. The namespace will be a sub-namespace inside the namespaces
            indicated in the package name of the proto file.
      **kwargs: Other arguments that will be passed to the cc_library.
    """
    if len(srcs) > 1:
        fail("Only one srcs value supported", "srcs")

    codegen_grpc_target = name + "__grpc_codegen"

    # TODO: Derive the output names here?
    # TODO: Support mock generation.
    flags = []
    if service_namespace:
        flags.append("services_namespace=" + service_namespace)

    generate_cc(
        well_known_protos = False,
        name = codegen_grpc_target,
        visibility = ["//visibility:private"],
        srcs = srcs,
        plugin = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
        flags = flags,
        **kwargs
    )

    features = ["-layering_check", "-parse_headers", "-no_undefined"]

    native.cc_library(
        name = name,
        srcs = [":" + codegen_grpc_target],
        hdrs = [":" + codegen_grpc_target],
        features = features,
        deps = ["@com_github_grpc_grpc//:grpc++_codegen_proto"] + deps,
        visibility = visibility,
        **kwargs
    )
