load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/*.rst",
        "**/*.yml",
    ]),
)

tensorstore_cc_library(
    name = "http",
    srcs = [
        "driver.cc",
    ],
    deps = [
        ":byte_range_util",
        "//tensorstore:context",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal:retries_context_resource",
        "//tensorstore/internal:retry",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":http",
        "//tensorstore:batch",
        "//tensorstore/internal:queue_testutil",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/http:mock_http_transport",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "byte_range_util",
    srcs = ["byte_range_util.cc"],
    hdrs = ["byte_range_util.h"],
    deps = [
        "//tensorstore/internal/http",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
    ],
)
