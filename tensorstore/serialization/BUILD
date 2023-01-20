load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "absl_time",
    srcs = ["absl_time.cc"],
    hdrs = ["absl_time.h"],
    deps = [
        ":serialization",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
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
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "batch",
    srcs = ["batch.cc"],
    hdrs = ["batch.h"],
    deps = [
        ":serialization",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
    ],
)

tensorstore_cc_library(
    name = "function",
    srcs = ["function.cc"],
    hdrs = ["function.h"],
    deps = [
        ":serialization",
        "//tensorstore/internal:heterogeneous_container",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
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
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/internal:json_fwd",
        "//tensorstore/internal:riegeli_json_input",
        "//tensorstore/internal:riegeli_json_output",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
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
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "protobuf",
    srcs = ["protobuf.cc"],
    hdrs = ["protobuf.h"],
    deps = [
        ":serialization",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:protobuf_lite",
        "@com_google_riegeli//riegeli/messages:message_parse",
        "@com_google_riegeli//riegeli/messages:message_serialize",
    ],
)

tensorstore_cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    deps = [
        ":serialization",
        "//tensorstore/internal:heterogeneous_container",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "result",
    hdrs = ["result.h"],
    deps = [
        ":serialization",
        ":status",
        "//tensorstore/internal:attributes",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "riegeli_delimited",
    srcs = ["riegeli_delimited.cc"],
    hdrs = ["riegeli_delimited.h"],
    deps = [
        "//tensorstore/internal:utf8",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/varint:varint_reading",
        "@com_google_riegeli//riegeli/varint:varint_writing",
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
        ":riegeli_delimited",
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:memory",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/poly",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/apply_members",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
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
        "@com_google_googletest//:gtest_main",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    deps = [
        ":serialization",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "status_test",
    srcs = ["status_test.cc"],
    deps = [
        ":serialization",
        ":status",
        ":test_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_googletest//:gtest",
    ],
)
