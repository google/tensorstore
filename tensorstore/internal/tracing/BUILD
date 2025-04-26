load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "span_attribute",
    hdrs = [
        "span_attribute.h",
    ],
)

TRACING_DEFINES = []

tensorstore_cc_library(
    name = "tracing",
    srcs = [
        "logged_trace_span.cc",
    ],
    hdrs = [
        "local_trace_span.h",
        "logged_trace_span.h",
        "operation_trace_span.h",
        "trace_context.h",
    ],
    defines = TRACING_DEFINES,
    deps = [
        ":span_attribute",
        "//tensorstore/internal:source_location",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/log:log_streamer",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "trace_test",
    srcs = ["trace_test.cc"],
    deps = [
        ":span_attribute",
        ":tracing",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/log:scoped_mock_log",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)
