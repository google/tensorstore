# TensorStore driver adapter that performs data type conversion

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
    name = "cast",
    srcs = ["cast.cc"],
    hdrs = ["cast.h"],
    deps = [
        "//tensorstore:data_type",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/driver",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:nditerable_data_type_conversion",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution:any_receiver",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "cast_test",
    size = "small",
    srcs = ["cast_test.cc"],
    deps = [
        ":cast",
        "//tensorstore:cast",
        "//tensorstore:context",
        "//tensorstore:open",
        "//tensorstore:spec",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/driver/array",
        "//tensorstore/driver/zarr",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
