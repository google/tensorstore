# OMETiff TensorStore driver

load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
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
    name = "ometiff",
    deps = [
            ":driver",
    ],
)

tensorstore_cc_library(
    name = "metadata",
    srcs = ["metadata.cc"],
    hdrs = ["metadata.h"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:codec_spec",
        "//tensorstore:data_type",
        "//tensorstore:schema",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:data_type_endian_conversion",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal:json_metadata_matching",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:data_type",
        "//tensorstore/internal/json_binding:dimension_indexed",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
    linkopts = ["/usr/lib/x86_64-linux-gnu//libtiffxx.so.5",
            "/usr/lib/x86_64-linux-gnu//libtiff.so",
            "/usr/lib/x86_64-linux-gnu//libdeflate.so",
            "/usr/lib/x86_64-linux-gnu//libdeflate.so.0",
            ]
)


tensorstore_cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    deps = [
        ":metadata",
        "//tensorstore",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:path",
        "//tensorstore/internal/cache:chunk_cache",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:future",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "metadata_test",
    size = "small",
    srcs = ["metadata_test.cc"],
    deps = [
        ":metadata",
        "//tensorstore:array",
        "//tensorstore:codec_spec",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":driver",
        "//tensorstore:open",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:parse_json_matches",
        "//tensorstore/internal/cache",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)
