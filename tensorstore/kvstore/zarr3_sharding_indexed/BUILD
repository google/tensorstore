# zarr v3 sharded_indexed format kvstore driver

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
    name = "zarr3_sharding_indexed",
    srcs = ["zarr3_sharding_indexed.cc"],
    hdrs = ["zarr3_sharding_indexed.h"],
    deps = [
        ":key",
        ":shard_format",
        "//tensorstore:batch",
        "//tensorstore:context",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:transaction",
        "//tensorstore/driver/zarr3/codec:codec_chain_spec",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache:kvs_backed_cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:dimension_indexed",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/serialization",
        "//tensorstore/util:bit_vec",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:flow_sender_operation_state",
        "//tensorstore/util/execution:result_sender",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "zarr3_sharding_indexed_test",
    size = "small",
    srcs = ["zarr3_sharding_indexed_test.cc"],
    deps = [
        ":key",
        ":zarr3_sharding_indexed",
        "//tensorstore:batch",
        "//tensorstore:context",
        "//tensorstore:index",
        "//tensorstore:transaction",
        "//tensorstore/driver/zarr3/codec:codec_chain_spec",
        "//tensorstore/driver/zarr3/codec:crc32c",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:kvs_backed_cache_testutil",
        "//tensorstore/internal/riegeli:digest_suffixed_writer",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/internal/thread:thread_pool",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:executor",
        "//tensorstore/util:extents",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_re2//:re2",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:write",
        "@com_google_riegeli//riegeli/digests:crc32c_digester",
    ],
)

tensorstore_cc_library(
    name = "key",
    srcs = ["key.cc"],
    hdrs = ["key.h"],
    deps = [
        "//tensorstore:contiguous_layout",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:extents",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "shard_format",
    srcs = ["shard_format.cc"],
    hdrs = ["shard_format.h"],
    deps = [
        ":key",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:static_cast",
        "//tensorstore/driver/zarr3/codec",
        "//tensorstore/driver/zarr3/codec:codec_chain_spec",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal:unowned_to_shared",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:extents",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:wrapping_writer",
        "@com_google_riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "key_test",
    srcs = ["key_test.cc"],
    deps = [
        ":key",
        "//tensorstore:index",
        "//tensorstore/kvstore:key_range",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "shard_format_test",
    srcs = ["shard_format_test.cc"],
    deps = [
        ":shard_format",
        "//tensorstore:index",
        "//tensorstore/driver/zarr3/codec:codec_chain_spec",
        "//tensorstore/driver/zarr3/codec:codec_test_util",
        "//tensorstore/driver/zarr3/codec:crc32c",
        "//tensorstore/driver/zarr3/codec:gzip",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)
