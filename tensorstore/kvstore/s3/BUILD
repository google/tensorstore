load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

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
    name = "s3_resource",
    srcs = ["s3_resource.cc"],
    hdrs = ["s3_resource.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:env",
        "//tensorstore/internal/http:curl_handle",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore/gcs:admission_queue",
        "//tensorstore/kvstore/gcs:rate_limiter",
        "//tensorstore/kvstore/gcs:scaling_rate_limiter",
        "//tensorstore/internal:retry",
        "//tensorstore/internal:retries_context_resource",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "s3_credential_provider",
    srcs = ["s3_credential_provider.cc"],
    hdrs = ["s3_credential_provider.h"],
    deps = [
        "//tensorstore/internal:env",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_handle",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "s3_credential_provider_test",
    size = "small",
    srcs = ["s3_credential_provider_test.cc"],
    deps = [
        ":s3_credential_provider",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal:test_util",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_handle",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)


tensorstore_cc_library(
    name = "s3",
    srcs = ["s3_key_value_store.cc", "object_metadata.cc"],
    hdrs = ["object_metadata.h"],
    deps = [
        "//tensorstore/kvstore",
        ":s3_resource",
        ":s3_request_builder",
        ":validate",
        "@com_google_absl//absl/strings",
        "//tensorstore/kvstore/gcs:validate",
        "//tensorstore/internal/http",
        "//tensorstore/internal:retry",
    ],
    alwayslink = 1,
)


tensorstore_cc_test(
    name = "object_metadata_test",
    size = "small",
    srcs = [
        "object_metadata_test.cc",
    ],
    deps = [
        ":s3",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)


tensorstore_cc_test(
    name = "s3_key_value_store_test",
    size = "small",
    srcs = ["s3_key_value_store_test.cc"],
    deps = [
        ":s3",
        "//tensorstore:context",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal:path",
        "//tensorstore/internal:schedule_at",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_handle",
        "//tensorstore/internal/http:curl_transport",
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
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)


tensorstore_cc_library(
    name = "s3_request_builder",
    srcs = [
        "s3_request_builder.cc",
    ],
    hdrs = [
        "s3_request_builder.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "//tensorstore/internal/http",
        "//tensorstore/internal:ascii_utils",
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal:path",
        ":validate"
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "s3_request_builder_test",
    size = "small",
    srcs = ["s3_request_builder_test.cc"],
    deps = [
        ":s3_request_builder",
        "@com_google_googletest//:gtest_main",
        "//tensorstore/util:status_testutil",
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
        "@com_google_re2//:re2",
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
