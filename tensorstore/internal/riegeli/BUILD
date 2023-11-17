# Riegeli integration

load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "array_endian_codec",
    srcs = ["array_endian_codec.cc"],
    hdrs = ["array_endian_codec.h"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal:unaligned_data_type_functions",
        "//tensorstore/internal/metrics",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:endian",
        "//tensorstore/util:extents",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_riegeli//riegeli/base:chain",
        "@com_google_riegeli//riegeli/bytes:copy_all",
        "@com_google_riegeli//riegeli/bytes:limiting_reader",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "array_endian_codec_test",
    size = "small",
    srcs = ["array_endian_codec_test.cc"],
    deps = [
        ":array_endian_codec",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/util:endian",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:cord_test_helpers",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/zlib:zlib_reader",
        "@com_google_riegeli//riegeli/zlib:zlib_writer",
    ],
)

tensorstore_cc_library(
    name = "delimited",
    srcs = ["delimited.cc"],
    hdrs = ["delimited.h"],
    deps = [
        "//tensorstore/internal:utf8",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/varint:varint_reading",
        "@com_google_riegeli//riegeli/varint:varint_writing",
    ],
)

tensorstore_cc_library(
    name = "digest_suffixed_reader",
    hdrs = ["digest_suffixed_reader.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_riegeli//riegeli/base:arithmetic",
        "@com_google_riegeli//riegeli/base:object",
        "@com_google_riegeli//riegeli/base:types",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:limiting_reader",
        "@com_google_riegeli//riegeli/bytes:read_all",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/digests:digesting_reader",
        "@com_google_riegeli//riegeli/endian:endian_reading",
    ],
)

tensorstore_cc_library(
    name = "digest_suffixed_writer",
    hdrs = ["digest_suffixed_writer.h"],
    deps = [
        "@com_google_riegeli//riegeli/digests:digesting_writer",
        "@com_google_riegeli//riegeli/endian:endian_writing",
    ],
)

tensorstore_cc_test(
    name = "digest_suffixed_test",
    size = "small",
    srcs = ["digest_suffixed_test.cc"],
    deps = [
        ":digest_suffixed_reader",
        ":digest_suffixed_writer",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
        "@com_google_riegeli//riegeli/bytes:write",
        "@com_google_riegeli//riegeli/digests:crc32c_digester",
        "@com_google_riegeli//riegeli/zlib:zlib_reader",
        "@com_google_riegeli//riegeli/zlib:zlib_writer",
    ],
)

tensorstore_cc_library(
    name = "find",
    srcs = ["find.cc"],
    hdrs = ["find.h"],
    deps = ["@com_google_riegeli//riegeli/bytes:reader"],
)

tensorstore_cc_test(
    name = "find_test",
    srcs = ["find_test.cc"],
    deps = [
        ":find",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:string_reader",
    ],
)

tensorstore_cc_library(
    name = "json_input",
    srcs = ["json_input.cc"],
    hdrs = ["json_input.h"],
    deps = [
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_riegeli//riegeli/bytes:reader",
    ],
)

tensorstore_cc_library(
    name = "json_output",
    srcs = ["json_output.cc"],
    hdrs = ["json_output.h"],
    deps = [
        "//tensorstore/internal:unowned_to_shared",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "json_io_test",
    size = "small",
    srcs = ["json_io_test.cc"],
    deps = [
        ":json_input",
        ":json_output",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
    ],
)
