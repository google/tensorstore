load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "blosc_compressor",
    srcs = ["blosc_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal:json",
        "//tensorstore/internal/compression:blosc_compressor",
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
        "//tensorstore/internal:json",
        "//tensorstore/internal/compression:bzip2_compressor",
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
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_gtest",
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
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_registry",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/compression:json_specified_compressor",
        "@com_google_absl//absl/strings",
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
        ":metadata",
        ":spec",
        ":zarr",
        "//tensorstore:context",
        "//tensorstore:open",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/internal:json",
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
        ":zarr",
        "//tensorstore:context",
        "//tensorstore:open",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/internal:cache",
        "//tensorstore/internal:decoded_matches",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:json",
        "//tensorstore/internal:parse_json_matches",
        "//tensorstore/internal/compression:blosc",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/kvstore:key_value_store_testutil",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "dtype",
    srcs = ["dtype.cc"],
    hdrs = ["dtype.h"],
    deps = [
        "//tensorstore:data_type",
        "//tensorstore/internal:json",
        "//tensorstore/util:endian",
        "//tensorstore/util:extents",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "@com_github_nlohmann_json//:nlohmann_json",
    ],
)

tensorstore_cc_test(
    name = "dtype_test",
    size = "small",
    srcs = ["dtype_test.cc"],
    deps = [
        ":dtype",
        ":metadata_testutil",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
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
        "//tensorstore:data_type",
        "//tensorstore/internal:container_to_shared",
        "//tensorstore/internal:data_type_endian_conversion",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal:json",
        "//tensorstore/util:endian",
        "//tensorstore/util:result",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "metadata_test",
    size = "small",
    srcs = ["metadata_test.cc"],
    deps = [
        ":metadata",
        ":metadata_testutil",
        "//tensorstore:array_testutil",
        "//tensorstore:index",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/strings",
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
        "//tensorstore:index",
        "//tensorstore/internal:json",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/types:optional",
    ],
)

tensorstore_cc_test(
    name = "spec_test",
    size = "small",
    srcs = ["spec_test.cc"],
    deps = [
        ":metadata",
        ":spec",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "zarr",
    srcs = ["driver.cc"],
    hdrs = ["driver_impl.h"],
    deps = [
        ":metadata",
        ":spec",
        "//tensorstore",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:cache_key",
        "//tensorstore/internal:chunk_cache",
        "//tensorstore/internal:json",
        "//tensorstore/internal:path",
        "//tensorstore/internal:type_traits",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/util:bit_span",
        "//tensorstore/util:future",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "zlib_compressor",
    srcs = ["zlib_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal:json",
        "//tensorstore/internal/compression:zlib_compressor",
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
