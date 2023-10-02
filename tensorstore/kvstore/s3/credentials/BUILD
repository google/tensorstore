load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "aws_credential_provider",
    srcs = [
        "aws_credential_provider.cc",
        "cached_credential_provider.cc",
        "chained_credential_provider.cc",
        "environment_credential_provider.cc",
        "file_credential_provider.cc",
        "ec2_credential_provider.cc"],
    hdrs = [
        "aws_credential_provider.h",
        "cached_credential_provider.h",
        "chained_credential_provider.h",
        "environment_credential_provider.h",
        "expiry_credential_provider.h",
        "file_credential_provider.h",
        "ec2_credential_provider.h"],
    deps = [
        "//tensorstore/internal:env",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/util:result",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        ],
)

tensorstore_cc_test(
    name = "environmental_credential_provider_test",
    srcs = ["environment_credential_provider_test.cc"],
    deps = [
        ":aws_credential_provider",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "expiry_credential_provider_test",
    srcs = ["expiry_credential_provider_test.cc"],
    deps = [
        ":aws_credential_provider",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)


tensorstore_cc_test(
    name = "file_credential_provider_test",
    srcs = ["file_credential_provider_test.cc"],
    deps = [
        ":aws_credential_provider",
        "//tensorstore/internal:path",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)


tensorstore_cc_test(
    name = "chained_credential_provider_test",
    srcs = ["chained_credential_provider_test.cc"],
    deps = [
        ":aws_credential_provider",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "cached_credential_provider_test",
    srcs = ["cached_credential_provider_test.cc"],
    deps = [
        ":aws_credential_provider",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
