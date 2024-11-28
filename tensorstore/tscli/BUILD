load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_test(
    name = "args_test",
    srcs = ["args_test.cc"],
    deps = [
        ":tsclilib",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "tsclilib",
    srcs = [
        "args.cc",
        "kvstore_copy.cc",
        "kvstore_list.cc",
        "ts_print_spec.cc",
        "ts_print_stats.cc",
        "ts_search.cc",
    ],
    hdrs = [
        "args.h",
        "cli.h",
    ],
    deps = [
        "//tensorstore",
        "//tensorstore:array_storage_statistics",
        "//tensorstore:box",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:path",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:division",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:iterate",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_re2//:re2",
    ],
)

tensorstore_cc_binary(
    name = "tscli",
    srcs = [
        "main.cc",
    ],
    deps = [
        ":tsclilib",
        "//tensorstore:context",
        "//tensorstore/driver:all_drivers",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/kvstore:all_drivers",
        "//tensorstore/util:json_absl_flag",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
    ],
)
