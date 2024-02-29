# Defines Python bindings for CoordinatorServer
#
# This will be moved to //python/tensorstore once OCDBT is included in
# the open source release.

load("//bazel:pybind11.bzl", "pybind11_cc_library")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

pybind11_cc_library(
    name = "python",
    srcs = ["bindings.cc"],
    tags = ["manual"],
    deps = [
        "//python/tensorstore:context",
        "//python/tensorstore:future",
        "//python/tensorstore:json_type_caster",
        "//python/tensorstore:kvstore",
        "//python/tensorstore:result_type_caster",
        "//python/tensorstore:status",
        "//python/tensorstore:tensorstore_module_components",
        "//tensorstore:context",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/kvstore/ocdbt:dump_util",
        "//tensorstore/kvstore/ocdbt/distributed:coordinator_server",
        "//tensorstore/kvstore/ocdbt/format:dump",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util/garbage_collection:json",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/strings:cord",
    ],
    alwayslink = True,
)

doctest_test(
    name = "doctest_test",
    srcs = glob([
        "*.cc",
    ]),
)
