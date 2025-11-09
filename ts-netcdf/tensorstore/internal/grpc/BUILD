load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "client_credentials",
    srcs = ["client_credentials.cc"],
    hdrs = ["client_credentials.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal/grpc/clientauth:authentication_strategy",
        "//tensorstore/internal/grpc/clientauth:channel_authentication",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/synchronization",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_test(
    name = "client_credentials_test",
    srcs = ["client_credentials_test.cc"],
    deps = [
        ":client_credentials",
        "//tensorstore:context",
        "//tensorstore/util:result",
        "@googletest//:gtest_main",
        "@grpc//:grpc++",
        "@grpc//:grpc_security_base",
    ],
)

tensorstore_cc_library(
    name = "logging_interceptor",
    srcs = ["logging_interceptor.cc"],
    hdrs = ["logging_interceptor.h"],
    deps = [
        ":utils",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/proto:proto_util",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@com_google_protobuf//:protobuf",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_library(
    name = "server_credentials",
    srcs = ["server_credentials.cc"],
    hdrs = ["server_credentials.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal/grpc/serverauth:default_strategy",
        "//tensorstore/internal/grpc/serverauth:strategy",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/synchronization",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_test(
    name = "server_credentials_test",
    srcs = ["server_credentials_test.cc"],
    deps = [
        ":server_credentials",
        "//tensorstore:context",
        "//tensorstore/util:result",
        "@googletest//:gtest_main",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//tensorstore/internal:source_location",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@grpc//:grpc++",
    ],
)

tensorstore_cc_library(
    name = "grpc_mock",
    testonly = 1,
    hdrs = ["grpc_mock.h"],
    deps = [
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest",
        "@grpc//:gpr",
        "@grpc//:grpc++",
        "@grpc//:grpc_security_base",
    ],
)

tensorstore_cc_library(
    name = "peer_address",
    srcs = ["peer_address.cc"],
    hdrs = ["peer_address.h"],
    deps = [
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@grpc//:grpc++",
        "@re2",
    ],
)
