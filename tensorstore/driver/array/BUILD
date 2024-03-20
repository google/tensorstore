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
        "//tensorstore:array_storage_statistics",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:container_kind",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore:rank",
        "//tensorstore:spec",
        "//tensorstore:strided_layout",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/driver:chunk",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:dimension_units",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transform_array_constraints",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:arena",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:lock_collection",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:array",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/garbage_collection",
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
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:chunk_layout",
        "//tensorstore:context",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore:progress",
        "//tensorstore:spec",
        "//tensorstore:strided_layout",
        "//tensorstore/driver",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:dimension_units",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:unit",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
