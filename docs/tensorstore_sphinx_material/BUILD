load("//docs/tensorstore_sphinx_material:nodejs_rules.bzl", "nodejs_binary")
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load(":icons.bzl", "icon_symlinks")

package(
    default_visibility = ["//docs:__subpackages__"],
)

licenses(["notice"])

ICON_PACKAGES = [
    "@npm//@fortawesome/fontawesome-free",
    "@npm//@mdi/svg",
    "@npm//@primer/octicons",
]

ICON_ALIASES = {
    "material": "@mdi/svg/svg/",
    "octicons": "@primer/octicons/build/svg/",
    "fontawesome": "@fortawesome/fontawesome-free/svgs/",
}

nodejs_binary(
    name = "build_css",
    srcs = ["build_css.js"],
    entry_point = ":build_css.js",
    deps = [
        "@npm//autoprefixer",
        "@npm//cssnano",
        "@npm//material-design-color",
        "@npm//material-shadows",
        "@npm//postcss",
        "@npm//sass",
    ] + ICON_PACKAGES,
)

genrule(
    name = "build_css_genrule",
    srcs = glob([
        "src/assets/stylesheets/**/*.scss",
    ]),
    outs = [
        "sphinx_material/static/stylesheets/main.css",
        "sphinx_material/static/stylesheets/main.css.map",
        "sphinx_material/static/stylesheets/palette.css",
        "sphinx_material/static/stylesheets/palette.css.map",
    ],
    cmd = "$(location :build_css) " +
          # Use custom encoding of ICON_ALIASES rather than JSON encoding to
          # work around https://github.com/bazelbuild/rules_nodejs/issues/2744
          ":".join(["%s=%s" % (k, v) for (k, v) in ICON_ALIASES.items()]) + " " +
          "$(location src/assets/stylesheets/main.scss) " +
          "$(location src/assets/stylesheets/palette.scss) " +
          "$(OUTS)" +
          select({
              "//docs:no_minify_setting": "",
              "//conditions:default": " --optimize",
          }),
    tags = ["no-sandbox"],
    tools = [
        ":build_css",
    ],
)

nodejs_binary(
    name = "build_js",
    srcs = ["build_js.js"] + glob([
        "src/assets/javascripts/**",
        "tsconfig.json",
        "typings/**/*.d.ts",
    ]),
    entry_point = ":build_js.js",
    deps = [
        "@npm//clipboard",
        "@npm//esbuild",
        "@npm//escape-html",
        "@npm//focus-visible",
        "@npm//preact",
        "@npm//rxjs",
    ],
)

genrule(
    name = "build_js_genrule",
    outs = [
        "sphinx_material/static/javascripts/bundle.js",
        "sphinx_material/static/javascripts/bundle.js.map",
    ],
    cmd = "./$(location :build_js) $(OUTS)" +
          select({
              "//docs:no_minify_setting": "",
              "//conditions:default": " --optimize",
          }),
    tags = ["no-sandbox"],
    tools = [":build_js"],
)

HTML_TEMPLATE_SRC_PREFIX = "src/"

HTML_TEMPLATE_SRCS = glob([HTML_TEMPLATE_SRC_PREFIX + "**/*.html"])

[
    copy_file(
        name = x + "_copy_rule",
        src = x,
        out = "sphinx_material/" + x[len(HTML_TEMPLATE_SRC_PREFIX):],
        allow_symlink = True,
    )
    for x in HTML_TEMPLATE_SRCS
]

HTML_TEMPLATES_MINIFIED = [
    "sphinx_material/" + x[len(HTML_TEMPLATE_SRC_PREFIX):]
    for x in HTML_TEMPLATE_SRCS
]

filegroup(
    name = "html_templates",
    srcs = HTML_TEMPLATES_MINIFIED,
)

icon_symlinks(
    name = "icons",
    aliases = {
        target_prefix: "sphinx_material/.icons/" + alias + "/"
        for alias, target_prefix in ICON_ALIASES.items()
    },
    deps = ICON_PACKAGES,
)

py_library(
    name = "tensorstore_sphinx_material",
    srcs = glob(["sphinx_material/**/*.py"]),
    data = [
        "sphinx_material/theme.conf",
        ":html_templates",
        ":icons",
        ":sphinx_material/static/javascripts/bundle.js",
        ":sphinx_material/static/javascripts/bundle.js.map",
        ":sphinx_material/static/stylesheets/main.css",
        ":sphinx_material/static/stylesheets/main.css.map",
        ":sphinx_material/static/stylesheets/palette.css",
        ":sphinx_material/static/stylesheets/palette.css.map",
    ],
    deps = [
        "@pypa_docutils//:docutils",
        "@pypa_markupsafe//:markupsafe",
        "@pypa_sphinx//:sphinx",
    ],
)
