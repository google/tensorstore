load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "access_token",
    hdrs = ["access_token.h"],
    deps = ["@com_google_absl//absl/time"],
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
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googleapis//google/iam/credentials/v1:credentials_cc_grpc",
        "@com_google_googleapis//google/iam/credentials/v1:credentials_cc_proto",
    ],
)

tensorstore_cc_library(
    name = "authentication_strategy",
    srcs = ["authentication_strategy.cc"],
    hdrs = ["authentication_strategy.h"],
    deps = [
        "//tensorstore/internal/os:file_util",
        "//tensorstore/util:future",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log:absl_log",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "channel_authentication",
    srcs = ["channel_authentication.cc"],
    hdrs = ["channel_authentication.h"],
    deps = [
        ":authentication_strategy",
        "//tensorstore/util:future",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

tensorstore_cc_library(
    name = "call_authentication",
    srcs = ["call_authentication.cc"],
    hdrs = ["call_authentication.h"],
    deps = [
        ":authentication_strategy",
        "//tensorstore/util:future",
        "@com_github_grpc_grpc//:grpc++",
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
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googleapis//google/iam/credentials/v1:credentials_cc_proto",
    ],
)

tensorstore_cc_test(
    name = "call_authentication_test",
    srcs = ["call_authentication_test.cc"],
    deps = [
        ":call_authentication",
        "//tensorstore/util:status_testutil",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "channel_authentication_test",
    srcs = ["channel_authentication_test.cc"],
    deps = [
        ":channel_authentication",
        "//tensorstore/util:status_testutil",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "create_channel",
    srcs = ["create_channel.cc"],
    hdrs = ["create_channel.h"],
    deps = [
        ":authentication_strategy",
        "//tensorstore/internal/grpc:logging_interceptor",
        "@com_github_grpc_grpc//:grpc++",
    ],
)
