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

tensorstore_cc_test(
    name = "localstack_test",
    size = "small",
    srcs = ["localstack_test.cc"],
    args = [
        "--localstack_binary=$(location :moto_server)",
        "--binary_mode=moto",
    ],
    data = [":moto_server"],
    flaky = 1,  # Spawning the test process can be flaky.
    tags = [
        "cpu:2",
        "requires-net:loopback",
        "skip-cmake",
    ],
    deps = [
        ":s3_context",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:env",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/http:transport_test_utils",
        "//tensorstore/internal/os:subprocess",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_github_aws_cpp_sdk//:core",
        "@com_github_aws_cpp_sdk//:s3",

    ],
)
