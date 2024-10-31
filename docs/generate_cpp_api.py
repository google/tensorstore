# Copyright 2022 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates the JSON description of the C++ API from the preprocesed headers.

This takes as input a single preprocessed C++ header file and uses the
sphinx-immaterial C++ API parser to generate a JSON API description, that is
then consumed during the documentation build by the sphinx-immaterial cpp apigen
extension.

The input is typically the `:cpp_api_preprocessed.cc` Bazel target, which is
generated from the `cpp_api_include.cc` header file.

To add additional entities to the API documentation, ensure that they are
transitively included by `cpp_api_include.cc` and not excluded by any the
configuration options below.

Entities are filtered by original source path and namespace.

While the sphinx-immaterial cpp apigen extension supports generating the JSON
API description during the documentation build, this separate script is used in
order to improve incremental build efficiency (the JSON API description does not
need to be rebuilt in some cases).

API changes or additions can often result in undefined reference warnings from
the Sphinx C++ domain.  There are 3 types of warnings:

- References to namespaces: these are always bogus and just due to a limitation
  in the Sphinx C++ domain. They should be listed in `nitpick_ignore` in
  `conf.py`.

- References to internal names (such as private class members), commonly via a
  return type or SFINAE condition, that should not be exposed in the
  documentation. The internal name should be renamed to match one of the
  exclusions specified below.

- References to names that should be exposed but cannot (currently) be resolved
  by the Sphinx C++ domain, e.g. nested type aliases like
  `std::remove_cvref_t<T>::X`. These warnings should be silenced by adding a:

    // NONITPICK: std::remove_cvref_t<T>::X

  comment in the source code. Due to how such comments are currently extracted,
  the comment must appear between the first and last token of the entity to
  which it applies. A common location is immediately after the template
  parameters.
"""

import argparse
import code
import json
import pathlib

from sphinx_immaterial.apidoc.cpp import api_parser


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--source", required=True)
  ap.add_argument("--flags-file", required=True)
  ap.add_argument("--output")
  ap.add_argument("--verbose", action="store_true")
  ap.add_argument("--interactive", action="store_true")

  args = ap.parse_args()

  if args.interactive:
    args.verbose = True

  config = api_parser.Config(  # type: ignore[wrong-arg-types]
      input_path=args.source,
      compiler_flags=json.loads(
          pathlib.Path(args.flags_file).read_text(encoding="utf-8")
      )
      + [
          # Due to https://github.com/bazelbuild/bazel/issues/14764, this is not
          # picked up from .bazelrc.
          "-std=c++17"
      ],
      include_directory_map={"./": ""},
      # Only entities whose original source path matches any of these regular
      # expressions are included in the API documentation.
      allow_paths=[
          "^tensorstore/.*",
      ],
      # Entities whose original source path matches any of these regular
      # expressions are excluded from the API documentation. This takes
      # precedence over `allow_paths`.
      disallow_paths=[
          "/internal/",
          "^tensorstore/util/execution/",
          r"^tensorstore/util/division\.h$",
          r"^tensorstore/util/constant_vector\.h$",
          r"^tensorstore/util/garbage_collection/",
          r"^tensorstore/serialization/",
          r"^tensorstore/util/apply_members/",
      ],
      # Namespace names (not fully qualified) matching any of these regular
      # expressions are excluded from the API documentation.
      disallow_namespaces=[
          "^internal($|_)",
          "^execution",
      ],
      # Macros matching any of these regular expressions are excluded from the
      # API documentation.
      disallow_macros=[
          "^TENSORSTORE_INTERNAL_",
      ],
      ignore_diagnostics=[
          "__builtin_",
      ],
      # Initializers of variables and variable templates that match any of these
      # regular expressions will be elided.
      hide_initializers=[
          r"^=\s*(?:(true|false)\s*$|\[)",
      ],
      # Return types, SFINAE terms, and initializer expressions that match any
      # of these regular expressions will be elided. Return types will be shown
      # as `auto`.
      hide_types=[
          r"(\b|_)internal(\b|_)",
          r"\bdecltype\b",
          r"\bpoly::",
          r"\bStaticCastTraitsType\b",
          r"\bDataTypeConversionTraits\b",
          r"Impl\b",
      ],
      # Specifies type substitutions.
      #
      # The key specifies the type as it appears in the source code, not
      # necessarily fully qualified. The value specifies the substitution to
      # display in the documentation.
      type_replacements={
          "absl::remove_cvref_t": "std::remove_cvref_t",
          "tensorstore::internal::type_identity_t": "std::type_identity_t",
          "internal::type_identity_t": "std::type_identity_t",
          "SourceLocation": "std::source_location",
          "tensorstore::SourceLocation": "std::source_location",
          "absl::weak_ordering": "std::weak_ordering",
      },
      verbose=args.verbose,
  )

  if args.interactive:

    extractor = api_parser.Extractor(config)
    generator = api_parser.JsonApiGenerator(extractor)

    ns = {
        "config": config,
        "extractor": extractor,
        "generator": generator,
        "api_parser": api_parser,
    }
    code.interact(local=ns)
    return

  output_json = api_parser.generate_output(config)

  if args.output:
    pathlib.Path(args.output).write_text(
        json.dumps(output_json), encoding="utf-8"
    )


if __name__ == "__main__":
  main()
