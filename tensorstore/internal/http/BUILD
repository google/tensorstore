load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

# Ensure that ws2_32.lib is linked for mingw.
_WS2_32_LINKOPTS = select({
    "@tensorstore//:compiler_mingw_gcc": [
        "-lws2_32",
    ],
    "//conditions:default": [],
})

tensorstore_cc_library(
    name = "curl_factory",
    srcs = [
        "curl_factory.cc",
        "curl_factory_hook.cc",
    ],
    hdrs = ["curl_factory.h"],
    deps = [
        ":curl_wrappers",
        "//tensorstore/internal:env",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/log:verbose_flag",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@se_curl//:curl",
    ],
)

tensorstore_cc_library(
    name = "curl_handle",
    srcs = ["curl_handle.cc"],
    hdrs = ["curl_handle.h"],
    deps = [
        ":curl_factory",
        ":curl_wrappers",
        "//tensorstore/internal:source_location",
        "@com_google_absl//absl/log:absl_check",
        "@se_curl//:curl",
    ],
)

tensorstore_cc_library(
    name = "curl_transport",
    srcs = ["curl_transport.cc"],
    hdrs = ["curl_transport.h"],
    deps = [
        ":curl_factory",
        ":curl_handle",
        ":curl_wrappers",
        ":http",
        "//tensorstore/internal:cord_util",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/thread",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@se_curl//:curl",
    ],
)

tensorstore_cc_library(
    name = "curl_wrappers",
    srcs = ["curl_wrappers.cc"],
    hdrs = ["curl_wrappers.h"],
    deps = [
        "//tensorstore/internal:source_location",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@se_curl//:curl",
    ],
)

tensorstore_cc_test(
    name = "curl_wrappers_test",
    size = "small",
    srcs = ["curl_wrappers_test.cc"],
    deps = [
        ":curl_wrappers",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "http",
    srcs = [
        "http_request.cc",
        "http_response.cc",
        "http_transport.cc",
    ],
    hdrs = [
        "http_request.h",
        "http_response.h",
        "http_transport.h",
    ],
    deps = [
        ":http_header",
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_re2//:re2",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
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
        "//tensorstore/kvstore:byte_range",
        "@com_google_absl//absl/time",
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
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "curl_transport_test",
    srcs = ["curl_transport_test.cc"],
    linkopts = _WS2_32_LINKOPTS,
    deps = [
        ":curl_transport",
        ":http",
        ":transport_test_utils",
        "//tensorstore/internal/thread",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "curl_transport_http2_test",
    srcs = ["curl_transport_http2_test.cc"],
    linkopts = _WS2_32_LINKOPTS,
    deps = [
        ":curl_transport",
        ":http",
        ":transport_test_utils",
        "//tensorstore/internal/thread",
        "//tensorstore/util:future",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@org_nghttp2//:nghttp2",
    ],
)

tensorstore_cc_library(
    name = "http_header",
    srcs = ["http_header.cc"],
    hdrs = ["http_header.h"],
    deps = [
        "//tensorstore/internal:uri_utils",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
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
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "transport_test_utils",
    testonly = 1,
    srcs = ["transport_test_utils.cc"],
    hdrs = ["transport_test_utils.h"],
    linkopts = _WS2_32_LINKOPTS,
    deps = [
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "mock_http_transport",
    testonly = True,
    srcs = ["mock_http_transport.cc"],
    hdrs = ["mock_http_transport.h"],
    deps = [
        ":http",
        "//tensorstore/util:result",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
