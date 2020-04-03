load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "curl",
    srcs = [
        "curl_handle.cc",
        "curl_request.cc",
        "curl_request_builder.cc",
    ],
    hdrs = [
        "curl_handle.h",
        "curl_request.h",
        "curl_request_builder.h",
    ],
    deps = [
        ":http_response",
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:logging",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:variant",
        "@se_haxx_curl//:curl",
    ],
)

tensorstore_cc_test(
    name = "curl_request_builder_test",
    size = "small",
    srcs = ["curl_request_builder_test.cc"],
    deps = [
        ":curl",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "http_response",
    srcs = ["http_response.cc"],
    hdrs = ["http_response.h"],
    deps = [
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "http_response_test",
    size = "small",
    srcs = [
        "http_response_test.cc",
    ],
    deps = [
        ":http_response",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)
