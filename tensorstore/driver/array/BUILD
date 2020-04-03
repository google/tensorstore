load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_array",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/internal:type_traits",
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
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/util:executor",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_googletest//:gtest_main",
    ],
)
