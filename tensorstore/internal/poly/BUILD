load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "poly",
    srcs = [
        "poly.cc",
        "poly_impl.h",
    ],
    hdrs = ["poly.h"],
)

tensorstore_cc_test(
    name = "poly_test",
    size = "small",
    srcs = ["poly_test.cc"],
    deps = [
        ":poly",
        "//tensorstore/util:result",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_googletest//:gtest_main",
    ],
)
