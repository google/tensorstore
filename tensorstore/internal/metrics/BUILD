load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

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
        "//tensorstore/internal:logging",
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
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/poly",
        "//tensorstore/util:assert_macros",
        "@com_google_absl//absl/container:flat_hash_map",
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

tensorstore_cc_library(
    name = "metric_impl",
    hdrs = ["metric_impl.h"],
    deps = [
        ":collect",
        ":metadata",
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:assert_macros",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "collect",
    srcs = ["collect.cc"],
    hdrs = ["collect.h"],
    deps = [
        ":metadata",
        "//tensorstore/util:str_cat",
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
