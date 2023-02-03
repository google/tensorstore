load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "curl_handle",
    srcs = ["curl_handle.cc"],
    hdrs = ["curl_handle.h"],
    deps = [
        "//tensorstore/internal:attributes",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@se_curl//:curl",
    ],
)

tensorstore_cc_library(
    name = "curl_transport",
    srcs = [
        "curl_request_initialize.cc",
        "curl_transport.cc",
    ],
    hdrs = ["curl_transport.h"],
    deps = [
        ":curl_handle",
        ":http",
        "//tensorstore/internal:cord_util",
        "//tensorstore/internal:env",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:thread",
        "//tensorstore/internal/metrics",
        "//tensorstore/util:future",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@se_curl//:curl",
    ],
)

tensorstore_cc_library(
    name = "http",
    srcs = [
        "http_request.cc",
        "http_response.cc",
    ],
    hdrs = [
        "http_request.h",
        "http_response.h",
        "http_transport.h",
    ],
    deps = [
        "//tensorstore/internal:path",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "http_request_test",
    size = "small",
    srcs = [
        "http_request_test.cc",
    ],
    deps = [
        ":http",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "http_response_test",
    size = "small",
    srcs = [
        "http_response_test.cc",
    ],
    deps = [
        ":http",
        "//tensorstore/util:status",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "curl_transport_test",
    srcs = ["curl_transport_test.cc"],
    deps = [
        ":curl_transport",
        ":http",
        ":transport_test_utils",
        "//tensorstore/internal:thread",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "curl_transport_http2_test",
    srcs = ["curl_transport_http2_test.cc"],
    deps = [
        ":curl_transport",
        ":http",
        ":transport_test_utils",
        "//tensorstore/internal:thread",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@org_nghttp2//:nghttp2",
    ],
)

tensorstore_cc_library(
    name = "http_header",
    srcs = ["http_header.cc"],
    hdrs = ["http_header.h"],
    deps = [
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@com_google_re2//:re2",
    ],
)

tensorstore_cc_test(
    name = "http_header_test",
    size = "small",
    srcs = ["http_header_test.cc"],
    deps = [
        ":http_header",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "transport_test_utils",
    testonly = 1,
    srcs = ["transport_test_utils.cc"],
    hdrs = ["transport_test_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)
