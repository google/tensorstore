load(
    "//bazel:tensorstore.bzl",
    "tensorstore_cc_binary",
    "tensorstore_cc_library",
    "tensorstore_cc_test",
)

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "schedule_at",
    srcs = ["schedule_at.cc"],
    hdrs = ["schedule_at.h"],
    deps = [
        ":thread",
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/internal/container:intrusive_red_black_tree",
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
        ":pool_impl",
        ":task",
        ":task_group_impl",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/util:executor",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

tensorstore_cc_library(
    name = "thread_pool_benchmark_inc",
    testonly = 1,
    textual_hdrs = ["thread_pool_benchmark.inc"],
    deps = [
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/util:executor",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/synchronization",
        "@com_google_benchmark//:benchmark",  # build_cleaner: keep
    ],
)

tensorstore_cc_binary(
    name = "thread_pool_benchmark",
    testonly = 1,
    srcs = ["thread_pool_benchmark.cc"],
    deps = [
        ":thread_pool",
        ":thread_pool_benchmark_inc",
        "@com_google_absl//absl/flags:commandlineflag",
        "@com_google_absl//absl/flags:reflection",
        "@com_google_benchmark//:benchmark_main",
    ],
)

tensorstore_cc_library(
    name = "thread_pool_test_inc",
    testonly = 1,
    textual_hdrs = ["thread_pool_test.inc"],
    deps = [
        "//tensorstore/util:executor",
        "@com_google_absl//absl/random",
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
        ":thread_pool_test_inc",
        "@com_google_absl//absl/flags:commandlineflag",
        "@com_google_absl//absl/flags:reflection",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "task",
    hdrs = ["task.h"],
    deps = [
        "//tensorstore/internal:attributes",
        "//tensorstore/internal/tracing",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "task_provider",
    hdrs = ["task_provider.h"],
    deps = ["//tensorstore/internal:intrusive_ptr"],
)

tensorstore_cc_library(
    name = "pool_impl",
    srcs = ["pool_impl.cc"],
    hdrs = ["pool_impl.h"],
    deps = [
        ":task_provider",
        ":thread",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/container:circular_queue",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "pool_impl_test",
    size = "small",
    srcs = ["pool_impl_test.cc"],
    deps = [
        ":pool_impl",
        ":task",
        ":task_provider",
        "//tensorstore/internal:intrusive_ptr",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "task_group_impl",
    srcs = ["task_group_impl.cc"],
    hdrs = ["task_group_impl.h"],
    deps = [
        ":pool_impl",
        ":task",
        ":task_provider",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/container:block_queue",
        "//tensorstore/internal/container:single_producer_queue",
        "//tensorstore/internal/metrics",
        "//tensorstore/util:span",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
