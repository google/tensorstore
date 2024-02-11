load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load(
    "//bazel:tensorstore.bzl",
    "tensorstore_cc_binary",
    "tensorstore_cc_library",
    "tensorstore_cc_proto_library",
    "tensorstore_cc_test",
    "tensorstore_proto_library",
)

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

bool_flag(
    name = "enable",
    build_setting_default = True,
)

config_setting(
    name = "enable_setting",
    flag_values = {
        ":enable": "True",
    },
    visibility = ["//visibility:private"],
)

tensorstore_proto_library(
    name = "metrics_proto",
    srcs = ["metrics.proto"],
)

tensorstore_cc_proto_library(
    name = "metrics_cc_proto",
    deps = [":metrics_proto"],
)

tensorstore_cc_library(
    name = "collect",
    srcs = ["collect.cc"],
    hdrs = ["collect.h"],
    deps = [
        ":metadata",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "collect_test",
    srcs = ["collect_test.cc"],
    deps = [
        ":collect",
        "//tensorstore/internal:json_gtest",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "metadata",
    srcs = ["metadata.cc"],
    hdrs = ["metadata.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "metadata_test",
    srcs = ["metadata_test.cc"],
    deps = [
        ":metadata",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "metric_hook",
    hdrs = ["metric_hook.h"],
    deps = ["//tensorstore/internal:type_traits"],
)

tensorstore_cc_library(
    name = "metric_impl",
    srcs = ["metric_impl.cc"],
    hdrs = ["metric_impl.h"],
    deps = [
        ":metadata",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "metrics",
    hdrs = [
        "counter.h",
        "gauge.h",
        "histogram.h",
        "value.h",
    ],
    defines = select({
        ":enable_setting": [],
        "//conditions:default": ["TENSORSTORE_METRICS_DISABLED"],
    }),
    deps = [
        ":collect",
        ":metadata",
        ":metric_impl",
        ":registry",
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "metrics_test",
    srcs = ["metrics_test.cc"],
    deps = [
        ":collect",
        ":metrics",
        ":registry",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_binary(
    name = "metrics_benchmark_test",
    testonly = 1,
    srcs = ["metrics_benchmark_test.cc"],
    deps = [
        ":metrics",
        ":registry",
        "//tensorstore/internal/thread:thread_pool",
        "//tensorstore/util:executor",
        "@com_google_absl//absl/synchronization",
        "@com_google_benchmark//:benchmark_main",
    ],
)

tensorstore_cc_library(
    name = "protobuf",
    srcs = ["protobuf.cc"],
    hdrs = ["protobuf.h"],
    deps = [
        ":collect",
        ":metrics_cc_proto",
        "//tensorstore/util:span",
        "@com_google_absl//absl/log:absl_log",
    ],
)

tensorstore_cc_library(
    name = "prometheus",
    srcs = ["prometheus.cc"],
    hdrs = ["prometheus.h"],
    deps = [
        ":collect",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/http",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "prometheus_test",
    srcs = ["prometheus_test.cc"],
    deps = [
        ":collect",
        ":prometheus",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "protobuf_test",
    srcs = ["protobuf_test.cc"],
    deps = [
        ":metrics",
        ":metrics_cc_proto",
        ":protobuf",
        ":registry",
        "//tensorstore/proto:protobuf_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    deps = [
        ":collect",
        ":metadata",
        ":metric_hook",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/poly",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "registry_test",
    srcs = ["registry_test.cc"],
    deps = [
        ":collect",
        ":registry",
        "@com_google_googletest//:gtest_main",
    ],
)
