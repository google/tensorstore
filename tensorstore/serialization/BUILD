load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "absl_time",
    srcs = ["absl_time.cc"],
    hdrs = ["absl_time.h"],
    deps = [
        ":serialization",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "absl_time_test",
    size = "small",
    srcs = ["absl_time_test.cc"],
    deps = [
        ":absl_time",
        ":serialization",
        ":test_util",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "batch",
    srcs = ["batch.cc"],
    hdrs = ["batch.h"],
    deps = [
        ":serialization",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_library(
    name = "function",
    srcs = ["function.cc"],
    hdrs = ["function.h"],
    deps = [
        ":serialization",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/container:heterogeneous_container",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "function_test",
    size = "small",
    srcs = ["function_test.cc"],
    deps = [
        ":function",
        ":serialization",
        ":test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "json",
    srcs = ["json.cc"],
    hdrs = [
        "json.h",
        "json_bindable.h",
    ],
    deps = [
        ":serialization",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/riegeli:json_input",
        "//tensorstore/internal/riegeli:json_output",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "json_test",
    size = "small",
    srcs = ["json_test.cc"],
    deps = [
        ":json",
        ":serialization",
        ":test_util",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "protobuf",
    srcs = ["protobuf.cc"],
    hdrs = ["protobuf.h"],
    deps = [
        ":serialization",
        "@abseil-cpp//absl/status",
        "@com_google_protobuf//:protobuf_lite",
        "@riegeli//riegeli/messages:message_parse",
        "@riegeli//riegeli/messages:message_serialize",
    ],
)

tensorstore_cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    deps = [
        ":serialization",
        "//tensorstore/internal/container:heterogeneous_container",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/log:absl_log",
    ],
)

tensorstore_cc_test(
    name = "registry_test",
    size = "small",
    srcs = ["registry_test.cc"],
    deps = [
        ":registry",
        ":serialization",
        ":test_util",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "result",
    hdrs = ["result.h"],
    deps = [
        ":serialization",
        ":status",
        "//tensorstore/util:result",
    ],
)

tensorstore_cc_test(
    name = "result_test",
    srcs = ["result_test.cc"],
    deps = [
        ":result",
        ":serialization",
        ":test_util",
        "//tensorstore/util:result",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "serialization",
    srcs = ["serialization.cc"],
    hdrs = [
        "fwd.h",
        "serialization.h",
        "span.h",
        "std_array.h",
        "std_complex.h",
        "std_map.h",
        "std_optional.h",
        "std_pair.h",
        "std_set.h",
        "std_tuple.h",
        "std_variant.h",
        "std_vector.h",
    ],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:memory",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/internal/poly",
        "//tensorstore/internal/riegeli:delimited",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/apply_members",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.cc"],
    deps = [
        ":serialization",
        ":test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "span_test",
    size = "small",
    srcs = ["span_test.cc"],
    deps = [
        ":batch",
        ":serialization",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    deps = [
        ":serialization",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "status_test",
    srcs = ["status_test.cc"],
    deps = [
        ":serialization",
        ":status",
        ":test_util",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    deps = [
        ":batch",
        ":serialization",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest",
    ],
)
