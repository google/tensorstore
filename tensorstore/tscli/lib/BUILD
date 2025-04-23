load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "glob_to_regex",
    srcs = ["glob_to_regex.cc"],
    hdrs = ["glob_to_regex.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tensorstore_cc_test(
    name = "glob_to_regex_test",
    srcs = ["glob_to_regex_test.cc"],
    deps = [
        ":glob_to_regex",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "kvstore_copy",
    srcs = ["kvstore_copy.cc"],
    hdrs = ["kvstore_copy.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "kvstore_list",
    srcs = ["kvstore_list.cc"],
    hdrs = ["kvstore_list.h"],
    deps = [
        ":glob_to_regex",
        "//tensorstore:context",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_re2//:re2",
    ],
)

tensorstore_cc_library(
    name = "ts_print_stats",
    srcs = ["ts_print_stats.cc"],
    hdrs = ["ts_print_stats.h"],
    deps = [
        "//tensorstore",
        "//tensorstore:array_storage_statistics",
        "//tensorstore:box",
        "//tensorstore:context",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:division",
        "//tensorstore/util:future",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "ts_print_spec",
    srcs = ["ts_print_spec.cc"],
    hdrs = ["ts_print_spec.h"],
    deps = [
        "//tensorstore",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/util:result",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "ts_search",
    srcs = ["ts_search.cc"],
    hdrs = ["ts_search.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/internal:path",
        "//tensorstore/kvstore",
        "//tensorstore/util:result",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
