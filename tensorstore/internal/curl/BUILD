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
    name = "curl_factory",
    srcs = ["curl_factory_hook.cc"],
    hdrs = ["curl_factory.h"],
    deps = [
        ":curl_wrappers",
        "//tensorstore/internal/os:file_util",
        "@abseil-cpp//absl/log:absl_check",
        "@boringssl//:crypto",
        "@curl",
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
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@curl",
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
        ":default_factory",
        "//tensorstore/internal:cord_util",
        "//tensorstore/internal:env",
        "//tensorstore/internal/container:circular_queue",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/internal/thread",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@curl",
    ],
    alwayslink = True,
)

tensorstore_cc_test(
    name = "curl_transport_test",
    srcs = ["curl_transport_test.cc"],
    linkopts = _WS2_32_LINKOPTS,
    deps = [
        ":curl_transport",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:transport_test_utils",
        "//tensorstore/internal/thread",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
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
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@curl",
    ],
)

tensorstore_cc_test(
    name = "curl_wrappers_test",
    size = "small",
    srcs = ["curl_wrappers_test.cc"],
    deps = [
        ":curl_wrappers",
        "@abseil-cpp//absl/status",
        "@curl",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "default_factory",
    srcs = ["default_factory.cc"],
    hdrs = ["default_factory.h"],
    deps = [
        ":curl_factory",
        ":curl_wrappers",
        "//tensorstore/internal:env",
        "//tensorstore/internal/log:verbose_flag",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@curl",
    ],
)

tensorstore_cc_test(
    name = "http2_test",
    srcs = ["http2_test.cc"],
    args = [
        "--test_httpserver_binary=$(location //tensorstore/internal/http/py:h2_server)",
    ],
    data = ["//tensorstore/internal/http/py:h2_server"],
    flaky = 1,
    tags = [
        "cpu:2",
        "requires-net:loopback",
        "skip-cmake",
        # NOTE: This test can fail on darwin/windows with the error:
        # CURL error Unsupported protocol: Received HTTP/0.9 when not allowed
        # Which might be a problem with the custom certificate.
        "skip-darwin",
        "skip-windows",
    ],
    deps = [
        ":curl_transport",
        ":default_factory",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:test_httpserver",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)
