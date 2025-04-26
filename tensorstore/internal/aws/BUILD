load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "aws_api",
    srcs = [
        "aws_api.cc",
    ],
    hdrs = [
        "aws_api.h",
    ],
    deps = [
        ":tls_ctx",
        "//tensorstore/internal/log:verbose_flag",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/debugging:leak_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
        "@aws_c_auth",
        "@aws_c_cal",
        "@aws_c_common",
        "@aws_c_http",
        "@aws_c_io",
    ],
)

tensorstore_cc_test(
    name = "aws_api_test",
    size = "small",
    srcs = ["aws_api_test.cc"],
    # args = ["--tensorstore_verbose_logging=aws"],
    deps = [
        ":aws_api",
        "@aws_c_common",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "aws_credentials",
    srcs = ["aws_credentials.cc"],
    hdrs = ["aws_credentials.h"],
    deps = [
        ":aws_api",
        ":string_view",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util:future",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@aws_c_auth",
        "@aws_c_common",
    ],
)

tensorstore_cc_library(
    name = "http_mocking",
    srcs = ["http_mocking.cc"],
    hdrs = ["http_mocking.h"],
    deps = [
        ":string_view",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/log:verbose_flag",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/debugging:leak_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@aws_c_auth",
        "@aws_c_common",
        "@aws_c_http",
    ],
)

tensorstore_cc_library(
    name = "string_view",
    hdrs = ["string_view.h"],
    deps = ["@aws_c_common"],
)

tensorstore_cc_library(
    name = "tls_ctx",
    srcs = ["tls_ctx.cc"],
    hdrs = ["tls_ctx.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "@aws_c_common",
        "@aws_c_io",
    ],
)
