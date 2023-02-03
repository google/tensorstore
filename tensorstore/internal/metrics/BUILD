load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_proto_library", "tensorstore_cc_test", "tensorstore_proto_library")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

licenses(["notice"])

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
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
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

cc_library(
    name = "metric_hook",
    hdrs = ["metric_hook.h"],
    deps = ["//tensorstore/internal:type_traits"],
)

tensorstore_cc_library(
    name = "metric_impl",
    hdrs = ["metric_impl.h"],
    deps = [
        ":collect",
        ":metadata",
        "//tensorstore/internal:type_traits",
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

cc_library(
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
