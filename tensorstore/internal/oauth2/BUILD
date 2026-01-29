load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

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
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
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
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
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
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@nlohmann_json//:json",
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
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:mock_http_transport",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "google_auth_test_utils",
    testonly = True,
    srcs = ["google_auth_test_utils.cc"],
    hdrs = ["google_auth_test_utils.h"],
    deps = [
        "//tensorstore/internal:env",
        "//tensorstore/internal/testing:scoped_directory",
    ],
)

tensorstore_cc_library(
    name = "google_service_account_auth_provider",
    srcs = ["google_service_account_auth_provider.cc"],
    hdrs = ["google_service_account_auth_provider.h"],
    deps = [
        ":oauth2",
        ":oauth_utils",
        "//tensorstore/internal/http",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
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
        "@abseil-cpp//absl/container:flat_hash_map",
        "@googletest//:gtest_main",
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
        "bearer_token.h",
        "fixed_token_auth_provider.h",
        "gce_auth_provider.h",
        "refreshable_auth_provider.h",
    ],
    deps = [
        ":oauth_utils",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "oauth2_auth_provider",
    srcs = ["oauth2_auth_provider.cc"],
    hdrs = ["oauth2_auth_provider.h"],
    deps = [
        ":oauth2",
        ":oauth_utils",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/http",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "oauth2_auth_provider_test",
    size = "small",
    srcs = ["oauth2_auth_provider_test.cc"],
    deps = [
        ":oauth2_auth_provider",
        "//tensorstore/internal/http",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
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
        "//tensorstore/internal/json",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@boringssl//:crypto",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "oauth_utils_test",
    size = "small",
    srcs = ["oauth_utils_test.cc"],
    deps = [
        ":fake_private_key",
        ":oauth_utils",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)
