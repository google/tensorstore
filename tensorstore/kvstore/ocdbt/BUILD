# Optionally-cooperative Distributed B+tree driver

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

# To enable debug checks, specify:
# bazel build --//tensorstore/kvstore/ocdbt:debug
bool_flag(
    name = "debug",
    build_setting_default = False,
)

config_setting(
    name = "debug_setting",
    flag_values = {
        ":debug": "True",
    },
    visibility = ["//visibility:private"],
)

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/*.rst",
        "**/*.yml",
    ]),
)

tensorstore_cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    deps = [
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:raw_bytes_hex",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_riegeli//riegeli/zstd:zstd_writer",
    ],
)

tensorstore_cc_library(
    name = "debug_defines",
    hdrs = ["debug_defines.h"],
    defines = select({
        ":debug_setting": ["TENSORSTORE_INTERNAL_OCDBT_DEBUG"],
        "//conditions:default": [],
    }),
)

tensorstore_cc_library(
    name = "ocdbt",
    srcs = ["driver.cc"],
    hdrs = ["driver.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":btree_writer",
        ":config",
        ":io_handle",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:open_mode",
        "//tensorstore:transaction",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json_fwd",
        "//tensorstore/internal:path",
        "//tensorstore/internal:ref_counted_string",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/ocdbt/distributed:btree_writer",
        "//tensorstore/kvstore/ocdbt/distributed:rpc_security",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/kvstore/ocdbt/io:io_handle_impl",
        "//tensorstore/kvstore/ocdbt/non_distributed:btree_writer",
        "//tensorstore/kvstore/ocdbt/non_distributed:list",
        "//tensorstore/kvstore/ocdbt/non_distributed:read",
        "//tensorstore/serialization",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
    alwayslink = True,
)

tensorstore_cc_library(
    name = "io_handle",
    srcs = ["io_handle.cc"],
    hdrs = ["io_handle.h"],
    deps = [
        ":config",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "flush_promise_test",
    size = "small",
    srcs = ["flush_promise_test.cc"],
    deps = [
        ":io_handle",
        "//tensorstore/util:future",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "btree_writer",
    hdrs = ["btree_writer.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:future",
        "@com_google_absl//absl/strings:cord",
    ],
)

tensorstore_cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":config",
        ":ocdbt",
        ":test_util",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:transaction",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:json_fwd",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/testing:dynamic",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":io_handle",
        ":ocdbt",
        "//tensorstore:context",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_binary(
    name = "dump",
    srcs = ["dump_main.cc"],
    deps = [
        ":dump_util",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:path",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:all_drivers",
        "//tensorstore/kvstore/ocdbt/format:dump",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:result",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
    ],
)

tensorstore_cc_library(
    name = "dump_util",
    srcs = ["dump_util.cc"],
    hdrs = ["dump_util.h"],
    deps = [
        ":config",
        "//tensorstore:context",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/kvstore/ocdbt/format:dump",
        "//tensorstore/kvstore/ocdbt/io:indirect_data_kvstore_driver",
        "//tensorstore/kvstore/ocdbt/io:io_handle_impl",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "zarr_test",
    size = "medium",
    srcs = ["zarr_test.cc"],
    deps = [
        ":ocdbt",
        "//tensorstore:context",
        "//tensorstore:spec",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/driver/zarr",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "read_version_test",
    size = "small",
    srcs = ["read_version_test.cc"],
    deps = [
        ":ocdbt",
        ":test_util",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/kvstore/ocdbt/non_distributed:create_new_manifest",
        "//tensorstore/kvstore/ocdbt/non_distributed:list_versions",
        "//tensorstore/kvstore/ocdbt/non_distributed:read_version",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)
