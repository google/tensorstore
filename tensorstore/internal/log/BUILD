load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "verbose_flag",
    srcs = ["verbose_flag.cc"],
    hdrs = ["verbose_flag.h"],
    deps = [
        "//tensorstore/internal:env",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "verbose_flag_test",
    srcs = ["verbose_flag_test.cc"],
    deps = [
        ":verbose_flag",
        "@abseil-cpp//absl/base:core_headers",
        "@googletest//:gtest_main",
    ],
)
