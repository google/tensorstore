load("//bazel:tensorstore.bzl", "tensorstore_cc_library")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "sha256",
    hdrs = ["sha256.h"],
    deps = ["@com_google_boringssl//:crypto"],
)
