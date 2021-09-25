workspace(
    name = "com_google_tensorstore",
)

load("//:external.bzl", "tensorstore_dependencies")

tensorstore_dependencies()

register_toolchains("@local_config_python//:py_toolchain")

load("//:external2.bzl", "tensorstore_dependencies2")

tensorstore_dependencies2()
