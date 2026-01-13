load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

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
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:optional",
        "@org_aomedia_avif//:avif",
        "@riegeli//riegeli/base:external_ref",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:string_reader",
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
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@riegeli//riegeli/bytes:limiting_reader",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/endian:endian_reading",
    ],
)

tensorstore_cc_test(
    name = "bmp_test",
    srcs = ["bmp_test.cc"],
    deps = [
        ":bmp",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:string_reader",
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
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@libjpeg_turbo//:jpeg",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "jpeg_test",
    srcs = ["jpeg_test.cc"],
    deps = [
        ":image",
        ":jpeg",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:string_reader",
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
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@libpng//:png",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:string_reader",
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
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@libtiff//:tiff",
        "@riegeli//riegeli/base:byte_fill",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
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
        "//tensorstore/internal:path",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:read_all",
        "@riegeli//riegeli/bytes:string_reader",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
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
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:read_all",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:writer",
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

tensorstore_cc_library(
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
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:optional",
        "@libwebp//:webp",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "webp_test",
    srcs = ["webp_test.cc"],
    deps = [
        ":image",
        ":webp",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:string_reader",
    ],
)
