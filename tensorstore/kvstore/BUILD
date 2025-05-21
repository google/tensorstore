load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

DRIVER_DOCS = [
    "file",
    "gcs",
    "http",
    "kvstack",
    "memory",
    "neuroglancer_uint64_sharded",
    "ocdbt",
    "s3",
    "tsgrpc",
    "zarr3_sharding_indexed",
    "zip",
]

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/*.rst",
        "**/*.yml",
    ]) + [
        "//tensorstore/kvstore/" + driver + ":doc_sources"
        for driver in DRIVER_DOCS
    ],
)

DRIVERS = DRIVER_DOCS + [
    "gcs_grpc",
    "gcs_http",
]

EXTRA_DRIVER_TARGETS = []

tensorstore_cc_library(
    name = "all_drivers",
    deps = [
        "//tensorstore/kvstore/" + driver
        for driver in DRIVERS
    ] + EXTRA_DRIVER_TARGETS,
    alwayslink = True,
)

tensorstore_cc_library(
    name = "byte_range",
    srcs = ["byte_range.cc"],
    hdrs = ["byte_range.h"],
    deps = [
        "//tensorstore/serialization",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
    ],
)

tensorstore_cc_test(
    name = "byte_range_test",
    size = "small",
    srcs = ["byte_range_test.cc"],
    deps = [
        ":byte_range",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "generation",
    srcs = ["generation.cc"],
    hdrs = ["generation.h"],
    deps = [
        "//tensorstore/serialization",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/util:endian",
        "//tensorstore/util:quote_string",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "generation_test",
    size = "small",
    srcs = ["generation_test.cc"],
    deps = [
        ":generation",
        "//tensorstore/serialization:test_util",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "key_range",
    srcs = ["key_range.cc"],
    hdrs = ["key_range.h"],
    deps = [
        "//tensorstore/internal:compare",
        "//tensorstore/util:quote_string",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:compare",
    ],
)

tensorstore_cc_test(
    name = "key_range_test",
    size = "small",
    srcs = ["key_range_test.cc"],
    deps = [
        ":key_range",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/types:compare",
        "@googletest//:gtest_main",
    ],
)

# To enable debug checks, specify:
# bazel build --//tensorstore/kvstore:transaction_debug
bool_flag(
    name = "transaction_debug",
    build_setting_default = False,
)

config_setting(
    name = "transaction_debug_setting",
    flag_values = {
        ":transaction_debug": "True",
    },
    visibility = ["//visibility:private"],
)

tensorstore_cc_library(
    name = "kvstore",
    srcs = [
        "kvstore.cc",
        "operations.cc",
        "read_result.cc",
        "spec.cc",
        "transaction.cc",
        "url_registry.cc",
    ],
    hdrs = [
        "driver.h",
        "kvstore.h",
        "operations.h",
        "read_modify_write.h",
        "read_result.h",
        "registry.h",
        "spec.h",
        "supported_features.h",
        "transaction.h",
        "url_registry.h",
    ],
    local_defines = select({
        ":transaction_debug_setting": ["TENSORSTORE_INTERNAL_KVSTORE_TRANSACTION_DEBUG"],
        "//conditions:default": [],
    }),
    deps = [
        ":byte_range",
        ":generation",
        ":key_range",
        "//tensorstore:batch",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options",
        "//tensorstore:open_mode",
        "//tensorstore:transaction",
        "//tensorstore/internal:compare",
        "//tensorstore/internal:context_binding",
        "//tensorstore/internal:driver_kind_registry",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json_registry",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:path",
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/container:intrusive_red_black_tree",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/serialization",
        "//tensorstore/serialization:registry",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:option",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:any_sender",
        "//tensorstore/util/execution:flow_sender_operation_state",
        "//tensorstore/util/execution:future_collecting_receiver",
        "//tensorstore/util/execution:future_sender",
        "//tensorstore/util/execution:sender",
        "//tensorstore/util/execution:sender_util",
        "//tensorstore/util/execution:sync_flow_sender",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:compare",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "kvstore_test",
    size = "small",
    srcs = ["kvstore_test.cc"],
    deps = [
        ":kvstore",
        "//tensorstore:context",
        "//tensorstore:transaction",
        "//tensorstore/driver/json",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "test_matchers",
    testonly = 1,
    hdrs = ["test_matchers.h"],
    deps = [
        ":generation",
        ":kvstore",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":auto_detect",
        ":batch_util",
        ":byte_range",
        ":generation",
        ":key_range",
        ":kvstore",
        ":test_matchers",
        "//tensorstore:batch",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options",
        "//tensorstore:open_mode",
        "//tensorstore:transaction",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/internal/testing:dynamic",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/internal/testing:random_seed",
        "//tensorstore/internal/thread",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender_testutil",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest",
        "@nlohmann_json//:json",
    ],
    alwayslink = True,
)

tensorstore_cc_library(
    name = "mock_kvstore",
    testonly = 1,
    srcs = ["mock_kvstore.cc"],
    hdrs = ["mock_kvstore.h"],
    deps = [
        ":batch_util",
        ":byte_range",
        ":generation",
        ":key_range",
        ":kvstore",
        "//tensorstore:batch",
        "//tensorstore:context",
        "//tensorstore:transaction",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:utf8",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/testing:queue_testutil",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@nlohmann_json//:json",
    ],
    alwayslink = True,
)

tensorstore_cc_test(
    name = "transaction_test",
    size = "small",
    srcs = ["transaction_test.cc"],
    deps = [
        ":byte_range",
        ":generation",
        ":key_range",
        ":kvstore",
        ":mock_kvstore",
        ":test_matchers",
        ":test_util",
        "//tensorstore:transaction",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_binary(
    name = "copy",
    srcs = ["copy.cc"],
    deps = [
        ":all_drivers",
        ":generation",
        ":kvstore",
        "//tensorstore/internal:path",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_binary(
    name = "live_kvstore_test",
    testonly = 1,
    srcs = ["live_kvstore_test.cc"],
    deps = [
        ":all_drivers",
        ":kvstore",
        ":test_util",
        "//tensorstore:context",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/kvstore/tsgrpc",  # Experimental
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",  # buildcleaner: keep
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "batch_util",
    hdrs = [
        "batch_util.h",
        "generic_coalescing_batch_util.h",
    ],
    deps = [
        ":byte_range",
        ":generation",
        ":kvstore",
        "//tensorstore:batch",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_library(
    name = "common_metrics",
    hdrs = ["common_metrics.h"],
    deps = [
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
    ],
)

tensorstore_cc_test(
    name = "all_kvstores_test",
    srcs = ["all_kvstores_test.cc"],
    deps = [
        ":all_drivers",  # buildcleaner: keep
        ":kvstore",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "auto_detect",
    srcs = ["auto_detect.cc"],
    hdrs = ["auto_detect.h"],
    deps = [
        ":kvstore",
        "//tensorstore:batch",
        "//tensorstore/internal:path",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "auto_detect_test",
    size = "small",
    srcs = ["auto_detect_test.cc"],
    deps = [
        ":auto_detect",
        ":byte_range",
        ":generation",
        ":kvstore",
        ":mock_kvstore",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:executor",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)
