# TensorStore driver adapter that performs data type conversion

load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "cast",
    srcs = ["cast.cc"],
    hdrs = ["cast.h"],
    deps = [
        "//tensorstore:data_type",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/driver",
        "//tensorstore/internal:json",
        "//tensorstore/internal:nditerable_data_type_conversion",
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
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
        "//tensorstore/driver/array",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/index_space:json",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
