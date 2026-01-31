load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "json_binding",
    srcs = ["std_variant.cc"],
    hdrs = [
        "box.h",
        "enum.h",
        "json_binding.h",
        "optional_object.h",
        "std_array.h",
        "std_optional.h",
        "std_tuple.h",
        "std_variant.h",
    ],
    deps = [
        ":bindable",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:rank",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json:array",
        "//tensorstore/internal/json:same",
        "//tensorstore/internal/json:value_as",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "absl_time",
    hdrs = ["absl_time.h"],
    deps = [
        ":bindable",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json:value_as",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "absl_time_test",
    srcs = ["absl_time_test.cc"],
    deps = [
        ":absl_time",
        ":bindable",
        ":gtest",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/json",
        "//tensorstore/internal/testing:json_gtest",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "array",
    hdrs = ["array.h"],
    deps = [
        ":bindable",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:static_cast",
        "//tensorstore/internal/json:array",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "array_test",
    srcs = ["array_test.cc"],
    deps = [
        ":array",
        ":gtest",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/json",
        "//tensorstore/internal/testing:json_gtest",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "bindable",
    hdrs = ["bindable.h"],
    deps = [
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "data_type",
    srcs = ["data_type.cc"],
    hdrs = ["data_type.h"],
    deps = [
        ":bindable",
        ":json_binding",
        "//tensorstore:data_type",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal/json",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "data_type_test",
    srcs = ["data_type_test.cc"],
    deps = [
        ":bindable",
        ":data_type",
        ":gtest",
        "//tensorstore:data_type",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "dimension_indexed",
    hdrs = ["dimension_indexed.h"],
    deps = [
        ":bindable",
        ":json_binding",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/internal:dimension_labels",
        "//tensorstore/internal/json",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "enum_test",
    srcs = ["enum_test.cc"],
    deps = [
        ":gtest",
        ":json_binding",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/json",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "gtest",
    testonly = 1,
    hdrs = ["gtest.h"],
    deps = [
        ":bindable",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@googletest//:gtest",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "json_binding_test",
    srcs = ["json_binding_test.cc"],
    deps = [
        ":bindable",
        ":gtest",
        ":json_binding",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/json",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "optional_object_test",
    srcs = ["optional_object_test.cc"],
    deps = [
        ":gtest",
        ":json_binding",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "rational",
    hdrs = ["rational.h"],
    deps = [
        ":bindable",
        ":json_binding",
        "//tensorstore/internal/json:value_as",
        "//tensorstore/util:rational",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "rational_test",
    srcs = ["rational_test.cc"],
    deps = [
        ":gtest",
        ":rational",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:rational",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "raw_bytes_hex",
    srcs = ["raw_bytes_hex.cc"],
    hdrs = ["raw_bytes_hex.h"],
    deps = [
        "//tensorstore:json_serialization_options_base",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "raw_bytes_hex_test",
    size = "small",
    srcs = ["raw_bytes_hex_test.cc"],
    deps = [
        ":gtest",
        ":json_binding",
        ":raw_bytes_hex",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "staleness_bound",
    srcs = ["staleness_bound.cc"],
    hdrs = ["staleness_bound.h"],
    deps = [
        ":bindable",
        ":json_binding",
        "//tensorstore:staleness_bound",
        "//tensorstore/internal/json:value_as",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "staleness_bound_test",
    srcs = ["staleness_bound_test.cc"],
    deps = [
        ":bindable",
        ":gtest",
        ":staleness_bound",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:staleness_bound",
        "//tensorstore/internal/testing:json_gtest",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "std_array_test",
    srcs = ["std_array_test.cc"],
    deps = [
        ":gtest",
        ":json_binding",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "std_optional_test",
    srcs = ["std_optional_test.cc"],
    deps = [
        ":bindable",
        ":gtest",
        ":json_binding",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "std_tuple_test",
    srcs = ["std_tuple_test.cc"],
    deps = [
        ":gtest",
        ":json_binding",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "std_variant_test",
    srcs = ["std_variant_test.cc"],
    deps = [
        ":bindable",
        ":gtest",
        ":json_binding",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "unit",
    srcs = ["unit.cc"],
    hdrs = ["unit.h"],
    deps = [
        ":bindable",
        ":json_binding",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/json:value_as",
        "//tensorstore/util:unit",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
    ],
)
