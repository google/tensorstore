load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "avif",
    srcs = [
        "avif_common.cc",
        "avif_common.h",
        "avif_reader.cc",
        "avif_writer.cc",
    ],
    hdrs = [
        "avif_reader.h",
        "avif_writer.h",
    ],
    deps = [
        ":image",
        "//tensorstore:data_type",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/types:optional",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@org_aomedia_avif//:avif",
    ],
)

tensorstore_cc_test(
    name = "avif_test",
    srcs = ["avif_test.cc"],
    deps = [
        ":avif",
        ":image",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:string_reader",
    ],
)

tensorstore_cc_library(
    name = "bmp",
    srcs = [
        "bmp_reader.cc",
    ],
    hdrs = [
        "bmp_reader.h",
    ],
    deps = [
        ":image",
        "//tensorstore:data_type",
        "//tensorstore/util:span",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_riegeli//riegeli/bytes:limiting_reader",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/endian:endian_reading",
    ],
)

tensorstore_cc_test(
    name = "bmp_test",
    srcs = ["bmp_test.cc"],
    deps = [
        ":bmp",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:string_reader",
    ],
)

tensorstore_cc_library(
    name = "jpeg",
    srcs = [
        "jpeg_common.cc",
        "jpeg_common.h",
        "jpeg_reader.cc",
        "jpeg_writer.cc",
    ],
    hdrs = [
        "jpeg_reader.h",
        "jpeg_writer.h",
    ],
    deps = [
        ":image",
        "//tensorstore:data_type",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@jpeg",
    ],
)

tensorstore_cc_test(
    name = "jpeg_test",
    srcs = ["jpeg_test.cc"],
    deps = [
        ":jpeg",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:string_reader",
    ],
)

tensorstore_cc_library(
    name = "png",
    srcs = [
        "png_reader.cc",
        "png_writer.cc",
    ],
    hdrs = [
        "png_reader.h",
        "png_writer.h",
    ],
    deps = [
        ":image",
        "//tensorstore:data_type",
        "//tensorstore/util:endian",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@png",
    ],
)

tensorstore_cc_test(
    name = "png_test",
    srcs = ["png_test.cc"],
    deps = [
        ":image",
        ":png",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:string_reader",
    ],
)

tensorstore_cc_library(
    name = "tiff",
    srcs = [
        "tiff_common.cc",
        "tiff_common.h",
        "tiff_reader.cc",
        "tiff_writer.cc",
    ],
    hdrs = [
        "tiff_reader.h",
        "tiff_writer.h",
    ],
    deps = [
        ":image",
        "//tensorstore:data_type",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@libtiff//:tiff",
    ],
)

tensorstore_cc_test(
    name = "tiff_test",
    srcs = ["tiff_test.cc"],
    args = [
        "--tensorstore_test_data_dir=" +
        package_name() + "/testdata",
    ],
    data = [":testdata"],
    deps = [
        ":image",
        ":tiff",
        "//tensorstore:data_type",
        "//tensorstore/internal:path",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:fd_reader",
        "@com_google_riegeli//riegeli/bytes:read_all",
        "@com_google_riegeli//riegeli/bytes:string_reader",
    ],
)

tensorstore_cc_library(
    name = "image",
    srcs = [
        "image_info.cc",
        "image_view.cc",
    ],
    hdrs = [
        "image_info.h",
        "image_reader.h",
        "image_view.h",
        "image_writer.h",
    ],
    deps = [
        "//tensorstore:data_type",
        "//tensorstore/util:span",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "image_reader_test",
    srcs = ["image_reader_test.cc"],
    args = [
        "--tensorstore_test_data_dir=" +
        package_name() + "/testdata",
    ],
    data = [":testdata"],
    deps = [
        ":avif",
        ":bmp",
        ":image",
        ":jpeg",
        ":png",
        ":tiff",
        ":webp",
        "//tensorstore:data_type",
        "//tensorstore/internal:path",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:fd_reader",
        "@com_google_riegeli//riegeli/bytes:read_all",
    ],
)

tensorstore_cc_test(
    name = "image_writer_test",
    srcs = ["image_writer_test.cc"],
    deps = [
        ":avif",
        ":image",
        ":jpeg",
        ":png",
        ":tiff",
        ":webp",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:writer",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob(
        include = [
            "testdata/**",
        ],
    ),
)

cc_library(
    name = "webp",
    srcs = [
        "webp_reader.cc",
        "webp_writer.cc",
    ],
    hdrs = [
        "webp_reader.h",
        "webp_writer.h",
    ],
    deps = [
        ":image",
        "//tensorstore:data_type",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@libwebp//:webp",
    ],
)

cc_test(
    name = "webp_test",
    srcs = ["webp_test.cc"],
    deps = [
        ":image",
        ":webp",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:string_reader",
    ],
)
