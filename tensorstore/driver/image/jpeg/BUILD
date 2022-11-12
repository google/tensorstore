load("//bazel:tensorstore.bzl", "tensorstore_cc_library")
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
    name = "jpeg",
    srcs = ["driver.cc"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/driver",
        "//tensorstore/driver/image:driver_impl",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/image",
        "//tensorstore/internal/image:jpeg",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
    ],
    alwayslink = True,
)
