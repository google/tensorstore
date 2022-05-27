load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

DOCTEST_SOURCES = glob([
    "**/*.rst",
    "**/*.yml",
])

doctest_test(
    name = "doctest_test",
    srcs = DOCTEST_SOURCES,
)

filegroup(
    name = "doc_sources",
    srcs = DOCTEST_SOURCES,
)

tensorstore_cc_library(
    name = "png",
    srcs = ["driver.cc"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/driver",
        "//tensorstore/driver/image:driver_impl",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:memory",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/compression:png",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:cord",
    ],
    alwayslink = True,
)
