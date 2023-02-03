load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "client_credentials",
    srcs = ["client_credentials.cc"],
    hdrs = ["client_credentials.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "client_credentials_test",
    srcs = ["client_credentials_test.cc"],
    deps = [
        ":client_credentials",
        "//tensorstore:context",
        "//tensorstore/util:result",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "server_credentials",
    srcs = ["server_credentials.cc"],
    hdrs = ["server_credentials.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "server_credentials_test",
    srcs = ["server_credentials_test.cc"],
    deps = [
        ":server_credentials",
        "//tensorstore:context",
        "//tensorstore/util:result",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "grpc_mock",
    testonly = 1,
    hdrs = ["grpc_mock.h"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_googletest//:gtest",
    ],
)
