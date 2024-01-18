load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = [],
)

tensorstore_cc_library(
    name = "virtual_chunked",
    srcs = ["virtual_chunked.cc"],
    hdrs = ["//tensorstore:virtual_chunked.h"],
    deps = [
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:context",
        "//tensorstore:staleness_bound",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/driver:chunk_cache_driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache:chunk_cache",
        "//tensorstore/kvstore:generation",
        "//tensorstore/serialization",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/serialization:function",
        "//tensorstore/util:executor",
        "//tensorstore/util:option",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "virtual_chunked_test",
    srcs = ["virtual_chunked_test.cc"],
    deps = [
        "//tensorstore:virtual_chunked",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:queue_testutil",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:generation_testutil",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)
