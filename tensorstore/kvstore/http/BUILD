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
    name = "http_resource",
    srcs = ["http_resource.cc"],
    hdrs = ["http_resource.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal:env",
        "//tensorstore/internal:retries_context_resource",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/rate_limiter:admission_queue",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_log",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "http",
    srcs = [
        "driver.cc",
    ],
    deps = [
        ":byte_range_util",
        ":http_resource",
        "//tensorstore:context",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:default_transport",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/internal/metrics:registration",
        "//tensorstore/internal/rate_limiter",
        "//tensorstore/internal/thread:schedule_at",
        "//tensorstore/internal/uri:ascii_set",
        "//tensorstore/internal/uri:parse",
        "//tensorstore/internal/uri:percent_coder",
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
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
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
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:default_transport",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/http:mock_http_transport",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
    ],
)
