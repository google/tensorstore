# TensorStore driver that performs format auto-detection

load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
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
    name = "auto",
    srcs = ["driver.cc"],
    deps = [
        "//tensorstore:context",
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore:spec",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:driver_kind_registry",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:auto_detect",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":auto",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:schema",
        "//tensorstore:spec",
        "//tensorstore:transaction",
        "//tensorstore/driver/zarr3",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/kvstore/ocdbt",
        "//tensorstore/kvstore/zip",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)
