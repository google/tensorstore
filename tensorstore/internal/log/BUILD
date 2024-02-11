load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "verbose_flag",
    srcs = ["verbose_flag.cc"],
    hdrs = ["verbose_flag.h"],
    deps = [
        "//tensorstore/internal:env",
        "//tensorstore/internal:no_destructor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "verbose_flag_test",
    srcs = ["verbose_flag_test.cc"],
    deps = [
        ":verbose_flag",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_googletest//:gtest_main",
    ],
)
