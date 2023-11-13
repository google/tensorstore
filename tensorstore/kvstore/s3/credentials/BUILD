load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "aws_credential_provider",
    srcs = [
        "aws_credential_provider.cc",
        "default_credential_provider.cc",
        "environment_credential_provider.cc",
        "file_credential_provider.cc",
        "ec2_credential_provider.cc"],
    hdrs = [
        "aws_credential_provider.h",
        "default_credential_provider.h",
        "environment_credential_provider.h",
        "file_credential_provider.h",
        "ec2_credential_provider.h"],
    deps = [
        "//tensorstore/internal:env",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        ],
)


tensorstore_cc_library(
  name = "test_utils",
  srcs = ["test_utils.cc"],
  hdrs = ["test_utils.h"],
  deps = [
    "//tensorstore/internal/http",
    "//tensorstore/util:str_cat",
    "//tensorstore/util:result",
    "@com_google_absl//absl/container:flat_hash_map",
  ]
)

tensorstore_cc_test(
    name = "default_credential_provider_test",
    size = "small",
    srcs = ["default_credential_provider_test.cc"],
    deps = [
        ":aws_credential_provider",
        ":test_utils",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)


tensorstore_cc_test(
    name = "environmental_credential_provider_test",
    size = "small",
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
    name = "file_credential_provider_test",
    size = "small",
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
    name = "ec2_credential_provider_test",
    size = "small",
    srcs = ["ec2_credential_provider_test.cc"],
    deps = [
        ":aws_credential_provider",
        ":test_utils",
        "//tensorstore/internal/http",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)