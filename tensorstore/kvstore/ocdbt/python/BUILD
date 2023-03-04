# Defines Python bindings for CoordinatorServer
#
# This will be moved to //python/tensorstore once OCDBT is included in
# the open source release.

load("//bazel:pybind11.bzl", "pybind11_cc_library")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

pybind11_cc_library(
    name = "python",
    srcs = ["bindings.cc"],
    tags = ["manual"],
    deps = [
        "//python/tensorstore:json_type_caster",
        "//python/tensorstore:result_type_caster",
        "//python/tensorstore:status",
        "//python/tensorstore:tensorstore_module_components",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/kvstore/ocdbt/distributed:coordinator_server",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)
