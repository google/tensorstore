load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "span_attribute",
    hdrs = [
        "span_attribute.h",
    ],
)

tensorstore_cc_library(
    name = "tracing",
    srcs = [
        "logged_trace_span.cc",
        "trace_span.cc",
    ],
    hdrs = [
        "logged_trace_span.h",
        "trace_context.h",
        "trace_span.h",
    ],
    deps = [
        ":span_attribute",
        "//tensorstore/internal:source_location",
        "//tensorstore/util:span",
        "@com_google_absl//absl/log:log_streamer",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "trace_test",
    srcs = ["trace_test.cc"],
    deps = [
        ":span_attribute",
        ":tracing",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/log:scoped_mock_log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
