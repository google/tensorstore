load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "apply_members",
    hdrs = [
        "apply_members.h",
        "std_array.h",
        "std_complex.h",
        "std_pair.h",
        "std_tuple.h",
    ],
    deps = ["@abseil-cpp//absl/base:core_headers"],
)

tensorstore_cc_test(
    name = "apply_members_test",
    size = "small",
    srcs = ["apply_members_test.cc"],
    deps = [
        ":apply_members",
        "@googletest//:gtest_main",
    ],
)
