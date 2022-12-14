load("//bazel:tensorstore.bzl", "tensorstore_cc_compile_test", "tensorstore_cc_library")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "cat",
    hdrs = ["cat.h"],
)

tensorstore_cc_compile_test(
    name = "cat_test",
    srcs = ["cat_test.cc"],
    deps = [
        ":cat",
        ":stringify",
    ],
)

tensorstore_cc_library(
    name = "defer",
    hdrs = ["defer.h"],
)

tensorstore_cc_library(
    name = "expand",
    hdrs = ["expand.h"],
)

tensorstore_cc_library(
    name = "stringify",
    hdrs = ["stringify.h"],
)

tensorstore_cc_compile_test(
    name = "stringify_test",
    srcs = ["stringify_test.cc"],
    deps = [":stringify"],
)
