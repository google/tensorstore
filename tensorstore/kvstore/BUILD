load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "generation",
    srcs = ["generation.cc"],
    hdrs = ["generation.h"],
    deps = [
        "//tensorstore/util:quote_string",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "generation_test",
    size = "small",
    srcs = ["generation_test.cc"],
    deps = [
        ":generation",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "generation_testutil",
    testonly = 1,
    hdrs = ["generation_testutil.h"],
    deps = [
        ":generation",
        "//tensorstore/util:result",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "byte_range",
    srcs = ["byte_range.cc"],
    hdrs = ["byte_range.h"],
    deps = [
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:to_string",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "byte_range_test",
    size = "small",
    srcs = ["byte_range_test.cc"],
    deps = [
        ":byte_range",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:to_string",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "key_value_store",
    srcs = ["key_value_store.cc"],
    hdrs = [
        "key_value_store.h",
        "registry.h",
    ],
    deps = [
        ":byte_range",
        ":generation",
        "//tensorstore:context",
        "//tensorstore:staleness_bound",
        "//tensorstore/internal:context_binding",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_registry",
        "//tensorstore/internal:logging",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:collecting_sender",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:sender",
        "//tensorstore/util:status",
        "//tensorstore/util:sync_flow_sender",
        "//tensorstore/util:to_string",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
)

tensorstore_cc_test(
    name = "key_value_store_test",
    size = "small",
    srcs = ["key_value_store_test.cc"],
    deps = [
        ":key_value_store",
        "//tensorstore:context",
        "//tensorstore/internal:json_object_with_type",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:to_string",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "key_value_store_testutil",
    testonly = 1,
    srcs = ["key_value_store_testutil.cc"],
    hdrs = ["key_value_store_testutil.h"],
    deps = [
        ":byte_range",
        ":generation",
        ":generation_testutil",
        ":key_value_store",
        "//tensorstore:context",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json_object_with_type",
        "//tensorstore/internal:logging",
        "//tensorstore/internal:queue_testutil",
        "//tensorstore/util:function_view",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
    ],
)
