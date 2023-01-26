workspace(
    name = "com_google_tensorstore",
)

load("//:external.bzl", "tensorstore_dependencies")

tensorstore_dependencies()

register_toolchains("@local_config_python//:py_toolchain")

load("@rules_perl//perl:deps.bzl", "perl_register_toolchains", "perl_rules_dependencies")

perl_rules_dependencies()
perl_register_toolchains()
