load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "metric_utils",
    srcs = ["metric_utils.cc"],
    hdrs = ["metric_utils.h"],
    deps = [
        "//tensorstore/internal/json",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/kvstore",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_binary(
    name = "ts_benchmark",
    testonly = True,
    srcs = ["ts_benchmark.cc"],
    deps = [
        ":metric_utils",
        "//tensorstore:all_drivers",
        "//tensorstore:context",
        "//tensorstore:index",
        "//tensorstore:spec",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/kvstore:all_drivers",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_binary(
    name = "kvstore_benchmark",
    srcs = ["kvstore_benchmark.cc"],
    deps = [
        ":metric_utils",
        "//tensorstore:context",
        "//tensorstore/internal:path",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:all_drivers",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:division",
        "//tensorstore/util:future",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_binary(
    name = "kvstore_duration",
    srcs = ["kvstore_duration.cc"],
    deps = [
        ":metric_utils",
        "//tensorstore:context",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:all_drivers",
        "//tensorstore/util:future",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
