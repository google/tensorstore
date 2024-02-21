load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "gcs_http",
    srcs = [
        "gcs_key_value_store.cc",
        "object_metadata.cc",
    ],
    hdrs = ["object_metadata.h"],
    deps = [
        ":gcs_resource",
        ":rate_limiter",
        "//tensorstore:context",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:env",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal:retries_context_resource",
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/oauth2",
        "//tensorstore/internal/oauth2:google_auth_provider",
        "//tensorstore/internal/thread:schedule_at",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/gcs:gcs_resource",
        "//tensorstore/kvstore/gcs:validate",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
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
    flaky = 1,  # This test has large timing variations, which can cause failures.
    deps = [
        ":gcs_http",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/oauth2:google_auth_provider",
        "//tensorstore/internal/oauth2:google_auth_test_utils",
        "//tensorstore/internal/thread:schedule_at",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:sender_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random:distributions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
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
        ":gcs_http",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:result",
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
        "//tensorstore/internal/container:intrusive_linked_list",
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
    name = "gcs_resource",
    srcs = ["gcs_resource.cc"],
    hdrs = ["gcs_resource.h"],
    deps = [
        ":admission_queue",
        ":rate_limiter",
        ":scaling_rate_limiter",
        "//tensorstore:context",
        "//tensorstore/internal:env",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "rate_limiter",
    srcs = ["rate_limiter.cc"],
    hdrs = ["rate_limiter.h"],
    deps = [
        "//tensorstore/internal/container:intrusive_linked_list",
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
        "//tensorstore/internal/poly",
        "//tensorstore/internal/thread:schedule_at",
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

tensorstore_cc_test(
    name = "gcs_testbench_test",
    size = "medium",
    srcs = [
        "gcs_testbench_test.cc",
    ],
    args = ["--testbench_binary=$(location @com_google_storagetestbench//:rest_server)"],
    data = ["@com_google_storagetestbench//:rest_server"],
    flaky = 1,
    tags = [
        "requires-net:loopback",
        "skip-cmake",  # storage_testbench is python, which doesn't work in bazel_to_cmake yet.
        "skip-darwin",  # the python binary doesn't find dependencies on darwin.
    ],
    deps = [
        ":gcs_http",
        "//tensorstore/internal:env",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/http:transport_test_utils",
        "//tensorstore/internal/thread",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/gcs:gcs_testbench",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)
