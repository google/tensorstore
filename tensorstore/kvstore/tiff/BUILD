load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "tiff_key_value_store",
    srcs = [
        "tiff_key_value_store.cc",
        "tiff_parser.cc",
    ],
    hdrs = [
        "tiff_key_value_store.h",
        "tiff_parser.h",
    ],
    deps = [
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
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:future",
        "@com_google_googletest//:gtest_main",
    ],
)
