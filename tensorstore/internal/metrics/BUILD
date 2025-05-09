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
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "collect_test",
    srcs = ["collect_test.cc"],
    deps = [
        ":collect",
        ":metadata",
        "//tensorstore/internal/testing:json_gtest",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "metadata",
    srcs = ["metadata.cc"],
    hdrs = ["metadata.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "metadata_test",
    srcs = ["metadata_test.cc"],
    deps = [
        ":metadata",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "metric_hook",
    hdrs = ["metric_hook.h"],
    deps = [
        ":metadata",
        "//tensorstore/internal/meta:type_traits",
    ],
)

tensorstore_cc_library(
    name = "metric_impl",
    srcs = ["metric_impl.cc"],
    hdrs = ["metric_impl.h"],
    deps = [
        ":metadata",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:node_hash_map",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "metrics",
    srcs = [
        "histogram.cc",
    ],
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
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/debugging:leak_check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "metrics_test",
    srcs = ["metrics_test.cc"],
    deps = [
        ":collect",
        ":metadata",
        ":metrics",
        ":registry",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_binary(
    name = "metrics_benchmark_test",
    testonly = 1,
    srcs = ["metrics_benchmark_test.cc"],
    deps = [
        ":metadata",
        ":metrics",
        ":registry",
        "//tensorstore/internal/thread:thread_pool",
        "//tensorstore/util:executor",
        "@abseil-cpp//absl/synchronization",
        "@google_benchmark//:benchmark_main",
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
        "@abseil-cpp//absl/log:absl_log",
    ],
)

tensorstore_cc_library(
    name = "prometheus",
    srcs = ["prometheus.cc"],
    hdrs = ["prometheus.h"],
    deps = [
        ":collect",
        ":metadata",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/http",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "prometheus_test",
    srcs = ["prometheus_test.cc"],
    deps = [
        ":collect",
        ":metadata",
        ":prometheus",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "protobuf_test",
    srcs = ["protobuf_test.cc"],
    deps = [
        ":collect",
        ":metadata",
        ":metrics",
        ":metrics_cc_proto",
        ":protobuf",
        ":registry",
        "//tensorstore/proto:protobuf_matchers",
        "@googletest//:gtest_main",
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
        "//tensorstore/internal/poly",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "registry_test",
    srcs = ["registry_test.cc"],
    deps = [
        ":collect",
        ":registry",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "histogram",
    srcs = ["histogram.cc"],
    hdrs = ["histogram.h"],
    deps = [
        ":collect",
        ":metadata",
        ":metric_impl",
        ":registry",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/debugging:leak_check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/strings",
    ],
)
