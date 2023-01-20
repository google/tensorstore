load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "fake_private_key",
    testonly = 1,
    srcs = ["fake_private_key.cc"],
    hdrs = ["fake_private_key.h"],
    deps = [],
)

tensorstore_cc_test(
    name = "fixed_token_auth_provider_test",
    size = "small",
    srcs = ["fixed_token_auth_provider_test.cc"],
    deps = [
        ":oauth2",
        "//tensorstore/util:result",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "gce_auth_provider_test",
    size = "small",
    srcs = ["gce_auth_provider_test.cc"],
    deps = [
        ":oauth2",
        "//tensorstore/internal:env",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "google_auth_provider",
    srcs = [
        "google_auth_provider.cc",
    ],
    hdrs = [
        "google_auth_provider.h",
    ],
    deps = [
        ":google_service_account_auth_provider",
        ":oauth2",
        ":oauth2_auth_provider",
        ":oauth_utils",
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
    name = "google_auth_provider_test",
    size = "small",
    srcs = ["google_auth_provider_test.cc"],
    deps = [
        ":fake_private_key",
        ":google_auth_provider",
        ":google_auth_test_utils",
        ":google_service_account_auth_provider",
        ":oauth2",
        ":oauth2_auth_provider",
        ":oauth_utils",
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
    name = "google_auth_test_utils",
    testonly = True,
    srcs = ["google_auth_test_utils.cc"],
    hdrs = ["google_auth_test_utils.h"],
    deps = [
        "//tensorstore/internal:env",
        "//tensorstore/internal:test_util",
    ],
)

tensorstore_cc_library(
    name = "google_service_account_auth_provider",
    srcs = ["google_service_account_auth_provider.cc"],
    hdrs = ["google_service_account_auth_provider.h"],
    deps = [
        ":oauth2",
        ":oauth_utils",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_handle",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "google_service_account_auth_provider_test",
    size = "small",
    srcs = ["google_service_account_auth_provider_test.cc"],
    deps = [
        ":fake_private_key",
        ":google_service_account_auth_provider",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "oauth2",
    srcs = [
        "auth_provider.cc",
        "fixed_token_auth_provider.cc",
        "gce_auth_provider.cc",
        "refreshable_auth_provider.cc",
    ],
    hdrs = [
        "auth_provider.h",
        "fixed_token_auth_provider.h",
        "gce_auth_provider.h",
        "refreshable_auth_provider.h",
    ],
    deps = [
        ":oauth_utils",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_handle",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "oauth2_auth_provider",
    srcs = ["oauth2_auth_provider.cc"],
    hdrs = ["oauth2_auth_provider.h"],
    deps = [
        ":oauth2",
        ":oauth_utils",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "oauth2_auth_provider_test",
    size = "small",
    srcs = ["oauth2_auth_provider_test.cc"],
    deps = [
        ":oauth2_auth_provider",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "oauth_utils",
    srcs = [
        "oauth_utils.cc",
    ],
    hdrs = [
        "oauth_utils.h",
    ],
    deps = [
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_boringssl//:crypto",
    ],
)

tensorstore_cc_test(
    name = "oauth_utils_test",
    size = "small",
    srcs = ["oauth_utils_test.cc"],
    deps = [
        ":fake_private_key",
        ":oauth_utils",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
