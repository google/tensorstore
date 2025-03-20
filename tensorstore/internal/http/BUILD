load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

# Ensure that ws2_32.lib is linked for mingw.
_WS2_32_LINKOPTS = select({
    "@tensorstore//:compiler_mingw_gcc": [
        "-DEFAULTLIB:ws2_32.lib",
        "-lws2_32",
    ],
    "@tensorstore//:compiler_msvc": ["-DEFAULTLIB:ws2_32.lib"],
    "//conditions:default": [],
})

tensorstore_cc_library(
    name = "default_transport",
    srcs = ["default_transport.cc"],
    hdrs = ["default_transport.h"],
    deps = [
        ":http",
        "//tensorstore/internal/curl:curl_transport",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/synchronization",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_re2//:re2",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
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
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:absl_check",
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
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_re2//:re2",
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
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:distributions",
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
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_boringssl//:crypto",
    ],
)

tensorstore_cc_test(
    name = "self_signed_cert_test",
    srcs = ["self_signed_cert_test.cc"],
    deps = [
        ":self_signed_cert",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
