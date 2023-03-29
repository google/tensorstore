load("//bazel:tensorstore.bzl", "tensorstore_cc_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_binary(
    name = "ts_benchmark",
    testonly = True,
    srcs = ["ts_benchmark.cc"],
    deps = [
        "//tensorstore:all_drivers",
        "//tensorstore:context",
        "//tensorstore:index",
        "//tensorstore:spec",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/internal:init_tensorstore",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/kvstore:all_drivers",
        "//tensorstore/kvstore/grpc:grpc_kvstore",  # Experimental
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_binary(
    name = "kvstore_benchmark",
    srcs = ["kvstore_benchmark.cc"],
    deps = [
        "//tensorstore:data_type",
        "//tensorstore/internal:init_tensorstore",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:all_drivers",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore/grpc:grpc_kvstore",  # Experimental
        "//tensorstore/util:division",
        "//tensorstore/util:future",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/flags:flag",
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
