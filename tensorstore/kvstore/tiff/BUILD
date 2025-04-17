load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "tiff_key_value_store",
    srcs = [
        "tiff_key_value_store.cc",
    ],
    hdrs = [
        "tiff_key_value_store.h",
    ],
    deps = [
        ":tiff_details",
        ":tiff_dir_cache",
        "//tensorstore/kvstore",
        "//tensorstore/util:future",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:cache_pool_resource",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_absl//absl/functional:function_ref",
    ],
)

tensorstore_cc_test(
    name = "tiff_key_value_store_test",
    srcs = ["tiff_key_value_store_test.cc"],
    deps = [
        ":tiff_key_value_store",
        ":tiff_test_util",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:future",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "tiff_dir_cache",
    srcs = ["tiff_dir_cache.cc"],
    hdrs = ["tiff_dir_cache.h"],
    deps = [
        ":tiff_details",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:cord",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/hash",
    ],
)

tensorstore_cc_test(
    name = "tiff_dir_cache_test",
    srcs = ["tiff_dir_cache_test.cc"],
    deps = [
        ":tiff_dir_cache",
        ":tiff_test_util",
        "//tensorstore:context",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore/memory",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:executor",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "tiff_details",
    srcs = ["tiff_details.cc"],
    hdrs = ["tiff_details.h"],
    deps = [
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/endian:endian_reading",
    ],
)

tensorstore_cc_test(
    name = "tiff_details_test",
    size = "small",
    srcs = ["tiff_details_test.cc"],
    deps = [
        ":tiff_details",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:string_reader",
    ],
)

tensorstore_cc_library(
    name = "tiff_test_util",
    srcs = ["tiff_test_util.cc"],
    hdrs = ["tiff_test_util.h"],
)
