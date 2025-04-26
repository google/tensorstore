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
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/internal/container:intrusive_red_black_tree",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/internal/os:fork_detection",
        "//tensorstore/internal/tracing",
        "//tensorstore/util:stop_token",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:compare",
    ],
)

tensorstore_cc_test(
    name = "schedule_at_test",
    size = "small",
    srcs = ["schedule_at_test.cc"],
    deps = [
        ":schedule_at",
        "//tensorstore/util:stop_token",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "thread",
    srcs = [
        "thread.cc",
    ],
    hdrs = ["thread.h"],
    deps = [
        "//tensorstore/internal/os:fork_detection",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log:absl_check",
    ],
)

tensorstore_cc_test(
    name = "thread_test",
    srcs = ["thread_test.cc"],
    deps = [
        ":thread",
        "@googletest//:gtest_main",
    ],
)

THREAD_POOL_DEFINES = []

THREAD_POOL_DEPS = []

tensorstore_cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    local_defines = THREAD_POOL_DEFINES,
    deps = THREAD_POOL_DEPS + [
        ":pool_impl",
        ":task",
        ":task_group_impl",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/os:fork_detection",
        "//tensorstore/internal/tracing",
        "//tensorstore/util:executor",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/log:absl_log",
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
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/synchronization",
        "@google_benchmark//:benchmark",  # build_cleaner: keep
    ],
)

tensorstore_cc_binary(
    name = "thread_pool_benchmark",
    testonly = 1,
    srcs = ["thread_pool_benchmark.cc"],
    deps = [
        ":thread_pool",
        ":thread_pool_benchmark_inc",
        "@abseil-cpp//absl/flags:commandlineflag",
        "@abseil-cpp//absl/flags:reflection",
        "@google_benchmark//:benchmark_main",
    ],
)

tensorstore_cc_library(
    name = "thread_pool_test_inc",
    testonly = 1,
    textual_hdrs = ["thread_pool_test.inc"],
    deps = [
        "//tensorstore/util:executor",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest",
    ],
)

tensorstore_cc_test(
    name = "thread_pool_test",
    size = "small",
    srcs = ["thread_pool_test.cc"],
    deps = [
        ":thread_pool",
        ":thread_pool_test_inc",
        "@abseil-cpp//absl/flags:commandlineflag",
        "@abseil-cpp//absl/flags:reflection",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "task",
    hdrs = ["task.h"],
    deps = [
        "//tensorstore/internal/tracing",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/time",
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
        "//tensorstore/internal/metrics:metadata",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
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
        "//tensorstore/internal/tracing",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/synchronization",
        "@googletest//:gtest_main",
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
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/internal/os:fork_detection",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
)
