# Filesystem-backed KeyValueStore driver

load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/*.rst",
        "**/*.yml",
    ]),
)

tensorstore_cc_library(
    name = "file",
    srcs = ["file_key_value_store.cc"],
    deps = [
        ":file_util",
        ":util",
        "//tensorstore:context",
        "//tensorstore/internal:cache_key",
        "//tensorstore/internal:context_binding",
        "//tensorstore/internal:file_io_concurrency_resource",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal:json",
        "//tensorstore/internal:os_error_code",
        "//tensorstore/internal:path",
        "//tensorstore/internal:poly",
        "//tensorstore/internal:type_traits",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/util:execution",
        "//tensorstore/util:executor",
        "//tensorstore/util:function_view",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:sender",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "file_key_value_store_test",
    size = "small",
    srcs = ["file_key_value_store_test.cc"],
    deps = [
        ":file",
        "//tensorstore:context",
        "//tensorstore/internal:file_io_concurrency_resource",
        "//tensorstore/internal:test_util",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:generation_testutil",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/kvstore:key_value_store_testutil",
        "//tensorstore/util:execution",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:sender_testutil",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "file_util",
    srcs = [
        "posix_file_util.cc",
        "windows_file_util.cc",
    ],
    hdrs = [
        "posix_file_util.h",
        "windows_file_util.h",
    ],
    deps = [
        ":util",
        "//tensorstore/internal:os_error_code",
        "//tensorstore/util:result",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_library(
    name = "util",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "unique_handle.h",
        "util.h",
    ],
    deps = [
        "//tensorstore/kvstore:key_range",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//tensorstore/kvstore:key_range",
        "@com_google_googletest//:gtest_main",
    ],
)
