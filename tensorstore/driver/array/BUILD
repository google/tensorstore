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
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    deps = [
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore/driver",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:dimension_units",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json:array",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:array",
        "//tensorstore/util/execution:any_receiver",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "array_test",
    size = "small",
    srcs = ["array_test.cc"],
    deps = [
        ":array",
        "//tensorstore",
        "//tensorstore:context",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:executor",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
