# Implementation of the I/O interface expected by Optionally-cooperative
# Distributed B+tree in terms of the TensorStore kvstore and cache facilities.

load("//bazel:tensorstore.bzl", "tensorstore_cc_library")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

cc_library(
    name = "coalesce_kvstore",
    srcs = ["coalesce_kvstore.cc"],
    hdrs = ["coalesce_kvstore.h"],
    deps = [
        "//tensorstore:transaction",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/ocdbt:debug_log",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "manifest_cache",
    srcs = ["manifest_cache.cc"],
    hdrs = ["manifest_cache.h"],
    deps = [
        "//tensorstore:transaction",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:kvs_backed_cache",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore/ocdbt:debug_log",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:cord",
    ],
)

tensorstore_cc_library(
    name = "node_cache",
    srcs = ["node_cache.cc"],
    hdrs = ["node_cache.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:kvs_backed_cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util/execution",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "io_handle_impl",
    srcs = ["io_handle_impl.cc"],
    hdrs = ["io_handle_impl.h"],
    deps = [
        ":coalesce_kvstore",
        ":indirect_data_kvstore_driver",
        ":indirect_data_writer",
        ":manifest_cache",
        ":node_cache",
        "//tensorstore:context",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore/ocdbt:config",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/kvstore/ocdbt/non_distributed:read_version",
        "//tensorstore/util:future",
        "//tensorstore/util:status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "indirect_data_writer",
    srcs = ["indirect_data_writer.cc"],
    hdrs = ["indirect_data_writer.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore/ocdbt:debug_log",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "indirect_data_kvstore_driver",
    srcs = ["indirect_data_kvstore_driver.cc"],
    hdrs = ["indirect_data_kvstore_driver.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore/ocdbt:debug_log",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_log",
    ],
)
