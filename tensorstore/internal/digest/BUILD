load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "sha256",
    srcs = ["sha256.cc"],
    hdrs = ["sha256.h"],
    deps = [
        "@com_google_absl//absl/strings:cord",
        "@com_google_boringssl//:crypto",
    ],
)

tensorstore_cc_test(
    name = "sha256_test",
    srcs = ["sha256_test.cc"],
    deps = [
        ":sha256",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
