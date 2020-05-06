load("//python/tensorstore:pytest.bzl", "tensorstore_pytest_test")
# Beam based tensorstore examples

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "run_pipeline",
    srcs = ["run_pipeline.py"],
    python_version = "PY3",
    tags = ["manual"],
    deps = [
        ":pipelines_lib",
        "@pypa_absl_py//:absl_py",
        "@pypa_apache_beam//:apache_beam",
        "@pypa_gin_config//:gin_config",
    ],
)

tensorstore_pytest_test(
    name = "reshard_tensor_test",
    size = "small",
    srcs = ["reshard_tensor_test.py"],
    tags = ["manual"],
    deps = [
        ":pipelines_lib",
        "//python/tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

py_library(
    name = "pipelines_lib",
    srcs = ["reshard_tensor.py"],
    tags = ["manual"],
    deps = [
        "//python/tensorstore",
        "@pypa_absl_py//:absl_py",
        "@pypa_apache_beam//:apache_beam",
        "@pypa_gin_config//:gin_config",
        "@pypa_numpy//:numpy",
    ],
)
