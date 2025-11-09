load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//visibility:public"])

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
    name = "zarr",
    deps = [
        ":blosc_compressor",
        ":bzip2_compressor",
        ":driver",
        ":zlib_compressor",
        ":zstd_compressor",
    ],
)

tensorstore_cc_library(
    name = "blosc_compressor",
    srcs = ["blosc_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal/compression:blosc_compressor",
        "//tensorstore/internal/json_binding",
        "@org_blosc_cblosc//:blosc",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "blosc_compressor_test",
    size = "small",
    srcs = ["blosc_compressor_test.cc"],
    deps = [
        ":blosc_compressor",
        ":compressor",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_library(
    name = "bzip2_compressor",
    srcs = ["bzip2_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal/compression:bzip2_compressor",
        "//tensorstore/internal/json_binding",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "bzip2_compressor_test",
    size = "small",
    srcs = ["bzip2_compressor_test.cc"],
    deps = [
        ":bzip2_compressor",
        ":compressor",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "compressor",
    srcs = ["compressor.cc"],
    hdrs = [
        "compressor.h",
        "compressor_registry.h",
    ],
    deps = [
        "//tensorstore/internal:json_registry",
        "//tensorstore/internal/compression:json_specified_compressor",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "@abseil-cpp//absl/base:no_destructor",
    ],
)

tensorstore_cc_test(
    name = "compressor_test",
    size = "small",
    srcs = ["compressor_test.cc"],
    deps = [
        ":compressor",
        ":zlib_compressor",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "driver_impl_test",
    size = "small",
    srcs = ["driver_impl_test.cc"],
    deps = [
        ":driver",
        ":metadata",
        ":spec",
        "//tensorstore",
        "//tensorstore:index",
        "//tensorstore:open",
        "//tensorstore:resize_options",
        "//tensorstore:transaction",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":driver",
        ":dtype",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:chunk_layout",
        "//tensorstore:context",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:rank",
        "//tensorstore:resize_options",
        "//tensorstore:schema",
        "//tensorstore:spec",
        "//tensorstore:staleness_bound",
        "//tensorstore:strided_layout",
        "//tensorstore:transaction",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/driver/n5",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/compression:blosc",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/internal/testing:decoded_matches",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/internal/testing:parse_json_matches",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "dtype",
    srcs = ["dtype.cc"],
    hdrs = ["dtype.h"],
    deps = [
        "//tensorstore:data_type",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:endian",
        "//tensorstore/util:extents",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "dtype_test",
    size = "small",
    srcs = ["dtype_test.cc"],
    deps = [
        ":dtype",
        ":metadata_testutil",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:endian",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "metadata",
    srcs = ["metadata.cc"],
    hdrs = ["metadata.h"],
    deps = [
        ":blosc_compressor",
        ":compressor",
        ":dtype",
        ":zlib_compressor",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/driver/zarr3:default_nan",
        "//tensorstore/internal:data_type_endian_conversion",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json:value_as",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:dimension_indexed",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/internal/riegeli:array_endian_codec",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:endian",
        "//tensorstore/util:extents",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@nlohmann_json//:json",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:read_all",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:write",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "metadata_test",
    size = "small",
    srcs = ["metadata_test.cc"],
    deps = [
        ":dtype",
        ":metadata",
        "//tensorstore:array",
        "//tensorstore:array_testutil",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:strided_layout",
        "//tensorstore/driver/zarr3:default_nan",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/util:endian",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "metadata_testutil",
    testonly = 1,
    srcs = ["metadata_testutil.cc"],
    hdrs = ["metadata_testutil.h"],
    deps = [
        ":metadata",
        "//tensorstore/internal/meta:type_traits",
        "@abseil-cpp//absl/meta:type_traits",
        "@googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "spec",
    srcs = ["spec.cc"],
    hdrs = ["spec.h"],
    deps = [
        ":compressor",
        ":dtype",
        ":metadata",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:schema",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:json_metadata_matching",
        "//tensorstore/internal/json:same",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:iterate",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
    ],
    alwayslink = True,
)

tensorstore_cc_test(
    name = "spec_test",
    size = "small",
    srcs = ["spec_test.cc"],
    deps = [
        ":dtype",
        ":metadata",
        ":spec",
        "//tensorstore:array",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:data_type",
        "//tensorstore:schema",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    hdrs = ["driver_impl.h"],
    deps = [
        ":metadata",
        ":spec",
        "//tensorstore:array",
        "//tensorstore:array_storage_statistics",
        "//tensorstore:box",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore:rank",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transform_broadcastable_array",
        "//tensorstore/internal:async_write_array",
        "//tensorstore/internal:chunk_grid_specification",
        "//tensorstore/internal:grid_storage_statistics",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:chunk_cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:auto_detect",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@nlohmann_json//:json",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "zlib_compressor",
    srcs = ["zlib_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal/compression:zlib_compressor",
        "//tensorstore/internal/json_binding",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "zlib_compressor_test",
    size = "small",
    srcs = ["zlib_compressor_test.cc"],
    deps = [
        ":compressor",
        ":zlib_compressor",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "zstd_compressor",
    srcs = ["zstd_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal/compression:zstd_compressor",
        "//tensorstore/internal/json_binding",
        "@riegeli//riegeli/zstd:zstd_writer",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "zstd_compressor_test",
    size = "small",
    srcs = ["zstd_compressor_test.cc"],
    deps = [
        ":compressor",
        ":zstd_compressor",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "storage_statistics_test_util",
    testonly = True,
    srcs = ["storage_statistics_test_util.cc"],
    hdrs = ["storage_statistics_test_util.h"],
    deps = [
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:array_storage_statistics",
        "//tensorstore:chunk_layout",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:schema",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@googletest//:gtest",
        "@nlohmann_json//:json",
    ],
    alwayslink = True,
)

tensorstore_cc_test(
    name = "storage_statistics_test",
    size = "small",
    srcs = ["storage_statistics_test.cc"],
    deps = [
        ":driver",
        ":storage_statistics_test_util",
        "@googletest//:gtest_main",
    ],
)
