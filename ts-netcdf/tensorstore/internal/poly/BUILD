load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "poly",
    srcs = [
        "poly_impl.h",
    ],
    hdrs = ["poly.h"],
    deps = [
        ":storage",
        "@abseil-cpp//absl/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "poly_test",
    size = "small",
    srcs = ["poly_test.cc"],
    deps = [
        ":poly",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/functional:function_ref",
        "@googletest//:gtest_main",
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
        "@googletest//:gtest_main",
    ],
)
