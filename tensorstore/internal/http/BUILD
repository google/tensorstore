load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "curl_handle",
    srcs = ["curl_handle.cc"],
    hdrs = ["curl_handle.h"],
    deps = [
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:logging",
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@se_haxx_curl//:curl",
    ],
)

tensorstore_cc_test(
    name = "curl_handle_test",
    srcs = ["curl_handle_test.cc"],
    deps = [
        ":curl_handle",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/internal:logging",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/util:future",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@se_haxx_curl//:curl",
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
        ":curl_handle",
        "//tensorstore/internal:logging",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:variant",
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
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "curl_transport_test",
    srcs = ["curl_transport_test.cc"],
    deps = [
        ":curl_transport",
        ":http",
        "//tensorstore/internal:logging",
        "//tensorstore/internal:mutex",
        "//tensorstore/util:executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@org_nghttp2//:nghttp2",
    ],
)
