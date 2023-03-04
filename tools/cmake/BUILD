load("//bazel:pytest.bzl", "tensorstore_pytest_test")

licenses(["notice"])

tensorstore_pytest_test(
    name = "bazel_to_cmake_test",
    size = "small",
    srcs = glob([
        "bazel_to_cmake.py",
        "bazel_to_cmake/**/*.py",
    ]),
    data = glob(["bazel_to_cmake/testdata/**"]),
)
