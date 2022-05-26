// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORSTORE_INTERNAL_JSON__PPRINT_PYTHON_H_
#define TENSORSTORE_INTERNAL_JSON__PPRINT_PYTHON_H_

/// \file
///
/// Pretty prints an `::nlohmnn::json` as a Python expression.
///
/// This is used to implement `tensorstore.Spec.__repr__` and
/// `tensorstore.TensorStore.__repr__`.
///
/// This is based on the `pprint.py` Python standard library module, but uses a
/// different output style that reduces nested indentation levels.

#include <string>
#include <string_view>

#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

struct PrettyPrintJsonAsPythonOptions {
  /// Additional indent to add for each nesting level.
  int indent = 2;
  /// Maximum line length.
  int width = 80;
  /// Number of initial characters to assume have already been written to the
  /// current line.  This effectively decreases the limit specified by `width`
  /// for the first line only.
  int cur_line_indent = 0;
  /// Number of spaces by which to indent subsequent lines.
  int subsequent_indent = 0;
};

void PrettyPrintJsonAsPython(
    std::string* out, const ::nlohmann::json& j,
    const PrettyPrintJsonAsPythonOptions& options = {});

std::string PrettyPrintJsonAsPython(
    const ::nlohmann::json& j,
    const PrettyPrintJsonAsPythonOptions& options = {});

/// Formats `prefix + *j + suffix`.
///
/// If `!j.ok()`, substitutes `"..."` for `*j`.
std::string PrettyPrintJsonAsPythonRepr(
    const Result<::nlohmann::json>& j, std::string_view prefix,
    std::string_view suffix,
    const PrettyPrintJsonAsPythonOptions& options = {});

}  // namespace internal_python
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON__PPRINT_PYTHON_H_
