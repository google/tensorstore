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
        "//tensorstore/internal:unaligned_data_type_functions",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:endian",
        "//tensorstore/util:extents",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@riegeli//riegeli/base:chain",
        "@riegeli//riegeli/base:external_ref",
        "@riegeli//riegeli/bytes:copy_all",
        "@riegeli//riegeli/bytes:limiting_reader",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
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
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:cord_test_helpers",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/zlib:zlib_reader",
        "@riegeli//riegeli/zlib:zlib_writer",
    ],
)

tensorstore_cc_library(
    name = "delimited",
    srcs = ["delimited.cc"],
    hdrs = ["delimited.h"],
    deps = [
        "//tensorstore/internal:utf8",
        "//tensorstore/util:quote_string",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/varint:varint_reading",
        "@riegeli//riegeli/varint:varint_writing",
    ],
)

tensorstore_cc_library(
    name = "digest_suffixed_reader",
    hdrs = ["digest_suffixed_reader.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:optional",
        "@riegeli//riegeli/base:arithmetic",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/base:object",
        "@riegeli//riegeli/base:types",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:limiting_reader",
        "@riegeli//riegeli/bytes:read_all",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/digests:digesting_reader",
        "@riegeli//riegeli/endian:endian_reading",
    ],
)

tensorstore_cc_library(
    name = "digest_suffixed_writer",
    hdrs = ["digest_suffixed_writer.h"],
    deps = [
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/digests:digesting_writer",
        "@riegeli//riegeli/endian:endian_writing",
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
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:read_all",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/bytes:write",
        "@riegeli//riegeli/digests:crc32c_digester",
        "@riegeli//riegeli/zlib:zlib_reader",
        "@riegeli//riegeli/zlib:zlib_writer",
    ],
)

tensorstore_cc_library(
    name = "find",
    srcs = ["find.cc"],
    hdrs = ["find.h"],
    deps = ["@riegeli//riegeli/bytes:reader"],
)

tensorstore_cc_test(
    name = "find_test",
    srcs = ["find_test.cc"],
    deps = [
        ":find",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:string_reader",
    ],
)

tensorstore_cc_library(
    name = "json_input",
    srcs = ["json_input.cc"],
    hdrs = ["json_input.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
        "@riegeli//riegeli/bytes:reader",
    ],
)

tensorstore_cc_library(
    name = "json_output",
    srcs = ["json_output.cc"],
    hdrs = ["json_output.h"],
    deps = [
        "//tensorstore/internal:unowned_to_shared",
        "@nlohmann_json//:json",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "json_io_test",
    size = "small",
    srcs = ["json_io_test.cc"],
    deps = [
        ":json_input",
        ":json_output",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
    ],
)
