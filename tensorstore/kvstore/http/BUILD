load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

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
        "//tensorstore:context",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:env",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal:retries_context_resource",
        "//tensorstore/internal:retry",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
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
        "//tensorstore/internal:queue_testutil",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
