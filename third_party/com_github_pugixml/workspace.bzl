load("//third_party:repo.bzl", "third_party_http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_github_pugixml",
        urls = [
            "http://github.com/zeux/pugixml/releases/download/v1.13/pugixml-1.13.zip",
        ],
        sha256 = "eb1334c98da79cd5c12930101cef31eaef1f5074729e2ac6af151a058b021a36",
        build_file = Label("//third_party:com_github_pugixml/bundled.BUILD.bazel"),
        system_build_file = Label("//third_party:com_github_pugixml/system.BUILD.bazel"),
        cmake_name = "pugixml",
        cmake_target_mapping = {
            "@com_github_pugixml//:pugixml": "pugixml::pugixml",
        },
        bazel_to_cmake = {},
    )
