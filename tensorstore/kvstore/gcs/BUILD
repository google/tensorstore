load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "gcs",
    srcs = [
        "gcs_key_value_store.cc",
        "object_metadata.cc",
    ],
    hdrs = ["object_metadata.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json",
        "//tensorstore/internal:path",
        "//tensorstore/internal:retry",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_handle",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/oauth2",
        "//tensorstore/internal/oauth2:google_auth_provider",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/util:execution",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "gcs_key_value_store_test",
    size = "small",
    srcs = [
        "gcs_key_value_store_test.cc",
        "gcs_mock.cc",
        "gcs_mock.h",
    ],
    deps = [
        ":gcs",
        "//tensorstore:context",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:logging",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_handle",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/kvstore:key_value_store_testutil",
        "//tensorstore/util:execution",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:sender_testutil",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "object_metadata_test",
    size = "small",
    srcs = [
        "object_metadata_test.cc",
    ],
    deps = [
        ":gcs",
        "//tensorstore/kvstore:key_value_store_testutil",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
