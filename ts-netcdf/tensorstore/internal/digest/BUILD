load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "sha256",
    srcs = ["sha256.cc"],
    hdrs = ["sha256.h"],
    deps = [
        "@abseil-cpp//absl/strings:cord",
        "@boringssl//:crypto",
    ],
)

tensorstore_cc_test(
    name = "sha256_test",
    srcs = ["sha256_test.cc"],
    deps = [
        ":sha256",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
    ],
)
