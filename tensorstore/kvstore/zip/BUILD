load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

DOCTEST_SOURCES = glob([
    "**/*.rst",
    "**/*.yml",
])

doctest_test(
    name = "doctest_test",
    srcs = DOCTEST_SOURCES,
)

filegroup(
    name = "doc_sources",
    srcs = DOCTEST_SOURCES,
)

tensorstore_cc_library(
    name = "zip",
    srcs = ["zip_key_value_store.cc"],
    deps = [
        ":cached_dir",
        ":zip_dir_cache",
        ":zip_write_cache",
        "//tensorstore:context",
        "//tensorstore:transaction",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/compression:zip_details",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/uri:parse",
        "//tensorstore/internal/uri:percent_coder",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:auto_detect",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:common_metrics",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/serialization",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender",
        "//tensorstore/util/execution:sender_util",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@riegeli//riegeli/bytes:cord_reader",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "zip_key_value_store_test",
    srcs = ["zip_key_value_store_test.cc"],
    args = [
        "--tensorstore_test_data=$(location //tensorstore/internal/compression:testdata/data.zip)",
    ],
    data = [
        "//tensorstore/internal/compression:testdata/data.zip",
        "//tensorstore/internal/compression:testdata/infozip_unicode_path.zip",
    ],
    deps = [
        ":zip",  # build_cleaner: keep
        "//tensorstore:context",
        "//tensorstore:transaction",
        "//tensorstore/internal/compression:zip_details",
        "//tensorstore/internal/compression:zip_easy",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/internal/thread",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/kvstore/ocdbt",  # build_cleaner: keep
        "//tensorstore/kvstore/zarr3_sharding_indexed",  # build_cleaner: keep
        "//tensorstore/util:endian",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender_testutil",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:fd_writer",
        "@riegeli//riegeli/bytes:read_all",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:write",
        "@riegeli//riegeli/digests:crc32_digester",
    ],
)

tensorstore_cc_library(
    name = "zip_dir_cache",
    srcs = ["zip_dir_cache.cc"],
    hdrs = ["zip_dir_cache.h"],
    deps = [
        ":cached_dir",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/compression:zip_details",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
    ],
)

tensorstore_cc_test(
    name = "zip_dir_cache_test",
    srcs = ["zip_dir_cache_test.cc"],
    args = [
        "--tensorstore_test_data=$(location //tensorstore/internal/compression:testdata/data.zip)",
    ],
    data = [
        "//tensorstore/internal/compression:testdata/data.zip",
        "//tensorstore/internal/compression:testdata/zip_test2.zip",
    ],
    deps = [
        ":zip_dir_cache",
        "//tensorstore:context",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:executor",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:read_all",
    ],
)

tensorstore_cc_test(
    name = "cached_dir_test",
    srcs = ["cached_dir_test.cc"],
    deps = [
        ":cached_dir",
        "//tensorstore/internal/compression:zip_details",
        "//tensorstore/internal/compression:zip_easy",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
    ],
)

tensorstore_cc_library(
    name = "cached_dir",
    srcs = ["cached_dir.cc"],
    hdrs = ["cached_dir.h"],
    deps = [
        "//tensorstore/internal/compression:zip_details",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@riegeli//riegeli/bytes:reader",
    ],
)

tensorstore_cc_library(
    name = "zip_write_cache",
    srcs = ["zip_write_cache.cc"],
    hdrs = ["zip_write_cache.h"],
    deps = [
        ":cached_dir",
        "//tensorstore:transaction",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:kvs_backed_cache",
        "//tensorstore/internal/compression:zip_details",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/digests:crc32_digester",
    ],
)

tensorstore_cc_test(
    name = "zip_write_cache_test",
    srcs = ["zip_write_cache_test.cc"],
    deps = [
        ":zip_write_cache",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/compression:zip_details",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:executor",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
    ],
)
