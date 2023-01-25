load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/*.rst",
        "**/*.yml",
    ]),
)

tensorstore_cc_library(
    name = "gcs",
    srcs = [
        "gcs_key_value_store.cc",
        "gcs_resource.cc",
        "gcs_resource.h",
        "object_metadata.cc",
    ],
    hdrs = ["object_metadata.h"],
    deps = [
        ":admission_queue",
        ":rate_limiter",
        ":scaling_rate_limiter",
        ":validate",
        "//tensorstore:context",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:env",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal:retries_context_resource",
        "//tensorstore/internal:retry",
        "//tensorstore/internal:schedule_at",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/oauth2",
        "//tensorstore/internal/oauth2:google_auth_provider",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
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
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:path",
        "//tensorstore/internal:schedule_at",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_handle",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/oauth2:google_auth_provider",
        "//tensorstore/internal/oauth2:google_auth_test_utils",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
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
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "admission_queue",
    srcs = ["admission_queue.cc"],
    hdrs = ["admission_queue.h"],
    deps = [
        ":rate_limiter",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "admission_queue_test",
    srcs = ["admission_queue_test.cc"],
    deps = [
        ":admission_queue",
        ":rate_limiter",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util:executor",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "rate_limiter",
    srcs = ["rate_limiter.cc"],
    hdrs = ["rate_limiter.h"],
    deps = [
        "//tensorstore/internal:intrusive_linked_list",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "scaling_rate_limiter",
    srcs = ["scaling_rate_limiter.cc"],
    hdrs = ["scaling_rate_limiter.h"],
    deps = [
        ":rate_limiter",
        "//tensorstore:data_type",
        "//tensorstore/internal:schedule_at",
        "//tensorstore/internal/poly",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "scaling_rate_limiter_test",
    srcs = ["scaling_rate_limiter_test.cc"],
    deps = [
        ":rate_limiter",
        ":scaling_rate_limiter",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util:executor",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "validate",
    srcs = [
        "validate.cc",
    ],
    hdrs = ["validate.h"],
    deps = [
        "//tensorstore/internal:utf8",
        "//tensorstore/kvstore:generation",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "validate_test",
    size = "small",
    srcs = [
        "validate_test.cc",
    ],
    deps = [
        ":validate",
        "@com_google_googletest//:gtest_main",
    ],
)
