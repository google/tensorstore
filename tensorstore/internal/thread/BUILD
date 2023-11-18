load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "schedule_at",
    srcs = ["schedule_at.cc"],
    hdrs = ["schedule_at.h"],
    deps = [
        ":thread",
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:intrusive_red_black_tree",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/tracing",
        "//tensorstore/util:stop_token",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "schedule_at_test",
    size = "small",
    srcs = ["schedule_at_test.cc"],
    deps = [
        ":schedule_at",
        "//tensorstore/util:stop_token",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "thread",
    srcs = ["thread.cc"],
    hdrs = ["thread.h"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_check",
    ],
)

tensorstore_cc_test(
    name = "thread_test",
    srcs = ["thread_test.cc"],
    deps = [
        ":thread",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        ":thread",
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/metrics",
        "//tensorstore/util:executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "thread_pool_test_cc",
    testonly = 1,
    textual_hdrs = ["thread_pool_test.inc"],
    deps = [
        "//tensorstore/util:executor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_test(
    name = "thread_pool_test",
    size = "small",
    srcs = ["thread_pool_test.cc"],
    deps = [
        ":thread_pool",
        ":thread_pool_test_cc",
        "@com_google_googletest//:gtest_main",
    ],
)
