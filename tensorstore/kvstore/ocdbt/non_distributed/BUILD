# Non-distributed operations for Optionally-cooperative Distributed B+tree

load("//bazel:tensorstore.bzl", "tensorstore_cc_library")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "read",
    srcs = ["read.cc"],
    hdrs = ["read.h"],
    deps = [
        ":storage_generation",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "read_version",
    srcs = ["read_version.cc"],
    hdrs = ["read_version.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "list",
    srcs = ["list.cc"],
    hdrs = ["list.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:flow_sender_operation_state",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "list_versions",
    srcs = ["list_versions.cc"],
    hdrs = ["list_versions.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:flow_sender_operation_state",
        "//tensorstore/util/execution:sync_flow_sender",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "create_new_manifest",
    srcs = ["create_new_manifest.cc"],
    hdrs = ["create_new_manifest.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore/ocdbt:config",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "btree_writer",
    srcs = ["btree_writer.cc"],
    hdrs = ["btree_writer.h"],
    deps = [
        ":create_new_manifest",
        ":list",
        ":staged_mutations",
        ":storage_generation",
        ":write_nodes",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/container:intrusive_red_black_tree",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/ocdbt:btree_writer",
        "//tensorstore/kvstore/ocdbt:config",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "staged_mutations",
    srcs = ["staged_mutations.cc"],
    hdrs = ["staged_mutations.h"],
    deps = [
        ":storage_generation",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/internal/container:intrusive_red_black_tree",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "write_nodes",
    srcs = ["write_nodes.cc"],
    hdrs = ["write_nodes.h"],
    deps = [
        "//tensorstore/kvstore/ocdbt:config",
        "//tensorstore/kvstore/ocdbt:io_handle",
        "//tensorstore/kvstore/ocdbt/format",
        "//tensorstore/util:result",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "storage_generation",
    srcs = ["storage_generation.cc"],
    hdrs = ["storage_generation.h"],
    deps = [
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore/ocdbt/format",
        "@blake3",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/strings:cord",
    ],
)
