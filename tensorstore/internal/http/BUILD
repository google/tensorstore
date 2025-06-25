load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

# Ensure that ws2_32.lib is linked for mingw/msvc.
_WS2_32_LINKOPTS = select({
    "@rules_cc//cc/compiler:mingw-gcc": [
        "-DEFAULTLIB:ws2_32.lib",
        "-lws2_32",
    ],
    "@rules_cc//cc/compiler:msvc-cl": [
        "-DEFAULTLIB:ws2_32.lib",
    ],
    "//conditions:default": [],
})

tensorstore_cc_library(
    name = "default_transport",
    srcs = ["default_transport.cc"],
    hdrs = ["default_transport.h"],
    deps = [
        ":http",
        "//tensorstore/internal/curl:curl_transport",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/synchronization",
    ],
    alwayslink = 1,
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
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@re2",
        "@riegeli//riegeli/bytes:cord_writer",
    ],
)

tensorstore_cc_library(
    name = "http_header",
    srcs = ["http_header.cc"],
    hdrs = ["http_header.h"],
    deps = [
        "//tensorstore/internal:ascii_set",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@re2",
    ],
)

tensorstore_cc_test(
    name = "http_header_test",
    size = "small",
    srcs = ["http_header_test.cc"],
    deps = [
        ":http_header",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
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
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
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
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "mock_http_transport",
    testonly = True,
    srcs = ["mock_http_transport.cc"],
    hdrs = ["mock_http_transport.h"],
    deps = [
        ":http",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "test_httpserver",
    testonly = True,
    srcs = ["test_httpserver.cc"],
    hdrs = ["test_httpserver.h"],
    deps = [
        ":self_signed_cert",
        ":transport_test_utils",
        "//tensorstore/internal:path",
        "//tensorstore/internal/os:file_util",
        "//tensorstore/internal/os:subprocess",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@re2",
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
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:distributions",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "self_signed_cert",
    srcs = ["self_signed_cert.cc"],
    hdrs = ["self_signed_cert.h"],
    deps = [
        "//tensorstore/internal:source_location",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@boringssl//:crypto",
    ],
)

tensorstore_cc_test(
    name = "self_signed_cert_test",
    srcs = ["self_signed_cert_test.cc"],
    deps = [
        ":self_signed_cert",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
    ],
)
