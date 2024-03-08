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
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/compression:json_specified_compressor",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
    ],
)

tensorstore_cc_test(
    name = "compressor_test",
    size = "small",
    srcs = ["compressor_test.cc"],
    deps = [
        ":compressor",
        ":zlib_compressor",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore:context",
        "//tensorstore:open",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/internal:decoded_matches",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:json_fwd",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal:parse_json_matches",
        "//tensorstore/internal/compression:blosc",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json_binding:gtest",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
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
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
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
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:endian",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json:value_as",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:dimension_indexed",
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
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:read_all",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:write",
        "@com_google_riegeli//riegeli/bytes:writer",
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
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "metadata_testutil",
    testonly = 1,
    srcs = ["metadata_testutil.cc"],
    hdrs = ["metadata_testutil.h"],
    deps = [
        ":metadata",
        "//tensorstore/internal:type_traits",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "spec",
    srcs = ["spec.cc"],
    hdrs = ["spec.h"],
    deps = [
        ":compressor",
        ":metadata",
        "//tensorstore:codec_spec",
        "//tensorstore:index",
        "//tensorstore:schema",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:json_metadata_matching",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
    ],
    alwayslink = True,
)

tensorstore_cc_test(
    name = "spec_test",
    size = "small",
    srcs = ["spec_test.cc"],
    deps = [
        ":metadata",
        ":spec",
        "//tensorstore:codec_spec",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    hdrs = ["driver_impl.h"],
    deps = [
        ":metadata",
        ":spec",
        "//tensorstore",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transform_broadcastable_array",
        "//tensorstore/internal:grid_storage_statistics",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/cache:chunk_cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:future",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/status",
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
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "zstd_compressor",
    srcs = ["zstd_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal/compression:zstd_compressor",
        "//tensorstore/internal/json_binding",
        "@com_google_riegeli//riegeli/zstd:zstd_writer",
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
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "storage_statistics_test_util",
    testonly = True,
    srcs = ["storage_statistics_test_util.cc"],
    hdrs = ["storage_statistics_test_util.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:open",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest",
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
        "@com_google_googletest//:gtest_main",
    ],
)
