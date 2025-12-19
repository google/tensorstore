load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "access_token",
    hdrs = ["access_token.h"],
    deps = ["@abseil-cpp//absl/time"],
)

tensorstore_cc_library(
    name = "iam_stub",
    srcs = ["iam_stub.cc"],
    hdrs = ["iam_stub.h"],
    deps = [
        ":access_token",
        ":authentication_strategy",
        ":create_channel",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/grpc:utils",
        "//tensorstore/proto:encode_time",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@googleapis//google/iam/credentials/v1:credentials_cc_grpc",
        "@googleapis//google/iam/credentials/v1:credentials_cc_proto",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_library(
    name = "authentication_strategy",
    srcs = ["authentication_strategy.cc"],
    hdrs = ["authentication_strategy.h"],
    deps = [
        "//tensorstore/internal/os:file_util",
        "//tensorstore/util:future",
        "@abseil-cpp//absl/log:absl_log",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_library(
    name = "access_token_cache",
    srcs = ["access_token_cache.cc"],
    hdrs = ["access_token_cache.h"],
    deps = [
        ":access_token",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_library(
    name = "channel_authentication",
    srcs = ["channel_authentication.cc"],
    hdrs = ["channel_authentication.h"],
    deps = [
        ":authentication_strategy",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/util:future",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_library(
    name = "call_authentication",
    srcs = ["call_authentication.cc"],
    hdrs = ["call_authentication.h"],
    deps = [
        ":authentication_strategy",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/util:future",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_library(
    name = "impersonate_service_account",
    srcs = ["impersonate_service_account.cc"],
    hdrs = ["impersonate_service_account.h"],
    deps = [
        ":access_token",
        ":access_token_cache",
        ":authentication_strategy",
        ":iam_stub",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@googleapis//google/iam/credentials/v1:credentials_cc_proto",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_test(
    name = "call_authentication_test",
    srcs = ["call_authentication_test.cc"],
    deps = [
        ":call_authentication",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_test(
    name = "channel_authentication_test",
    srcs = ["channel_authentication_test.cc"],
    deps = [
        ":channel_authentication",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_library(
    name = "create_channel",
    srcs = ["create_channel.cc"],
    hdrs = ["create_channel.h"],
    deps = [
        ":authentication_strategy",
        "//tensorstore/internal/grpc:logging_interceptor",
        "@grpc//:grpc++",
    ],
)
