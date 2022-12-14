load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "poly",
    srcs = [
        "poly_impl.h",
    ],
    hdrs = ["poly.h"],
    deps = [":storage"],
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

tensorstore_cc_library(
    name = "storage",
    srcs = ["storage.cc"],
    hdrs = ["storage.h"],
)

tensorstore_cc_test(
    name = "storage_test",
    srcs = ["storage_test.cc"],
    deps = [
        ":storage",
        "@com_google_googletest//:gtest_main",
    ],
)
