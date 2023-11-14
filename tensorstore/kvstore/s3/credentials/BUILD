load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "aws_credential_provider",
    hdrs = ["aws_credential_provider.h"],
)

tensorstore_cc_library(
  name = "environment_credential_provider",
  hdrs = ["environment_credential_provider.h"],
  srcs = ["environment_credential_provider.cc"],
  deps = [
    ":aws_credential_provider",
    "//tensorstore/internal:env",
    "//tensorstore/util:result",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/status",
  ]
)

tensorstore_cc_library(
  name = "file_credential_provider",
  hdrs = ["file_credential_provider.h"],
  srcs = ["file_credential_provider.cc"],
  deps = [
    ":aws_credential_provider",
    "//tensorstore/internal:env",
    "//tensorstore/internal:path",
    "//tensorstore/util:result",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/status",
  ]
)

tensorstore_cc_library(
  name = "ec2_credential_provider",
  hdrs = ["ec2_credential_provider.h"],
  srcs = ["ec2_credential_provider.cc"],
  deps = [
    ":aws_credential_provider",
    "//tensorstore/internal:env",
    "//tensorstore/util:result",
    "@com_google_absl//absl/log:absl_log",
    "@com_google_absl//absl/status",
    "//tensorstore/internal/http",
    "//tensorstore/internal/http:curl_transport",
    "//tensorstore/internal/json",
    "//tensorstore/internal/json_binding",
    "//tensorstore/internal/json_binding:absl_time",
    "//tensorstore/internal/json_binding:bindable",
  ]
)

tensorstore_cc_library(
  name = "default_credential_provider",
  srcs = ["default_credential_provider.cc"],
  hdrs = ["default_credential_provider.h"],
    deps = [
        ":aws_credential_provider",
        ":ec2_credential_provider",
        ":environment_credential_provider",
        ":file_credential_provider",
        "//tensorstore/internal:no_destructor",
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
  ],
  testonly = 1,
)

tensorstore_cc_test(
    name = "default_credential_provider_test",
    size = "small",
    srcs = ["default_credential_provider_test.cc"],
    deps = [
        ":default_credential_provider",
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
        ":environment_credential_provider",
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
        ":file_credential_provider",
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
        ":ec2_credential_provider",
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