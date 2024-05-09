# Placeholder: load py_binary
load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "*.rst",
        "*.yml",
    ]),
)
tensorstore_cc_library(
    name = "s3_context",
    srcs = ["s3_context.cc"],
    hdrs = ["s3_context.h"],
    deps = [
        "//tensorstore/util:executor",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/thread:thread_pool",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/synchronization",
        "@com_github_aws_cpp_sdk//:core",
    ]
)

tensorstore_cc_test(
    name = "s3_context_test",
    size = "small",
    srcs = ["s3_context_test.cc"],
    deps = [
        ":s3_context",
        "@com_github_aws_cpp_sdk//:s3",
        "@com_google_googletest//:gtest_main",
    ]
)

py_binary(
    name = "moto_server",
    testonly = 1,
    srcs = ["moto_server.py"],
    tags = [
        "manual",
        "notap",
        "skip-cmake",
    ],
    deps = ["@pypa_moto//:moto"],
)
