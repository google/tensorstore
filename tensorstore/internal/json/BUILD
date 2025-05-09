load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "json",
    srcs = ["json.cc"],
    hdrs = ["json.h"],
    deps = [
        ":value_as",
        "//tensorstore:index",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "value_as",
    srcs = ["value_as.cc"],
    hdrs = ["value_as.h"],
    deps = [
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "pprint_python",
    srcs = ["pprint_python.cc"],
    hdrs = ["pprint_python.h"],
    deps = [
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "array",
    srcs = ["array.cc"],
    hdrs = ["array.h"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:static_cast",
        "//tensorstore:strided_layout",
        "//tensorstore/internal:element_copy_function",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "pprint_python_test",
    size = "small",
    srcs = ["pprint_python_test.cc"],
    deps = [
        ":pprint_python",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "array_test",
    size = "small",
    srcs = ["array_test.cc"],
    deps = [
        ":array",
        ":value_as",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "json_test",
    size = "small",
    srcs = ["json_test.cc"],
    deps = [
        ":json",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "value_as_test",
    size = "small",
    srcs = ["value_as_test.cc"],
    deps = [
        ":value_as",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "same",
    srcs = ["same.cc"],
    hdrs = ["same.h"],
    deps = [
        "@abseil-cpp//absl/container:inlined_vector",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "same_test",
    size = "small",
    srcs = ["same_test.cc"],
    deps = [
        ":same",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)
