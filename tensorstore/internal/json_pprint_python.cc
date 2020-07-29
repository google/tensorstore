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

#include "tensorstore/internal/json_pprint_python.h"

#include <string>

#include "absl/strings/escaping.h"
#include <nlohmann/json.hpp>
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

namespace {

void FormatStringForPython(std::string* out, const std::string& s) {
  *out += '\'';
  *out += absl::CHexEscape(s);
  *out += '\'';
}

void FormatAsSingleLineForPython(std::string* out, const ::nlohmann::json& j) {
  switch (j.type()) {
    case ::nlohmann::json::value_t::object: {
      *out += "{";
      bool first = true;
      for (const auto& [key, value] :
           j.get_ref<const ::nlohmann::json::object_t&>()) {
        if (!first) {
          *out += ", ";
        } else {
          first = false;
        }
        FormatStringForPython(out, key);
        *out += ": ";
        FormatAsSingleLineForPython(out, value);
      }
      *out += "}";
      break;
    }
    case ::nlohmann::json::value_t::array: {
      *out += '[';
      bool first = true;
      for (const auto& x : j.get_ref<const ::nlohmann::json::array_t&>()) {
        if (!first) {
          *out += ", ";
        } else {
          first = false;
        }
        FormatAsSingleLineForPython(out, x);
      }
      *out += ']';
      break;
    }
    case ::nlohmann::json::value_t::string: {
      FormatStringForPython(out, j.get_ref<const std::string&>());
      break;
    }
    case ::nlohmann::json::value_t::boolean: {
      *out += (j.get_ref<const bool&>() ? "True" : "False");
      break;
    }
    case ::nlohmann::json::value_t::null: {
      *out += "None";
      break;
    }
    default:
      *out += j.dump();
      break;
  }
}

void PrettyPrintJsonObjectAsPythonInternal(
    std::string* out, const ::nlohmann::json::object_t& obj,
    PrettyPrintJsonAsPythonOptions options) {
  *out += '{';
  for (const auto& [key, value] : obj) {
    *out += '\n';
    auto new_options = options;
    new_options.subsequent_indent += options.indent;
    new_options.cur_line_indent = new_options.subsequent_indent;
    new_options.width -= 1;
    out->append(new_options.subsequent_indent, ' ');
    std::size_t prev_size = out->size();
    FormatStringForPython(out, key);
    std::size_t key_repr_len = out->size() - prev_size;
    *out += ": ";
    new_options.cur_line_indent += key_repr_len + 2;
    PrettyPrintJsonAsPython(out, value, new_options);
    *out += ',';
  }
  if (!obj.empty()) {
    *out += '\n';
    out->append(options.subsequent_indent, ' ');
  }
  *out += '}';
}

void PrettyPrintJsonArrayAsPythonInternal(
    std::string* out, const ::nlohmann::json::array_t& arr,
    PrettyPrintJsonAsPythonOptions options) {
  *out += '[';
  auto new_options = options;
  new_options.subsequent_indent += options.indent;
  new_options.cur_line_indent = new_options.subsequent_indent;
  new_options.width -= 1;
  for (const auto& value : arr) {
    *out += '\n';
    out->append(new_options.subsequent_indent, ' ');
    PrettyPrintJsonAsPython(out, value, new_options);
    *out += ',';
  }
  if (!arr.empty()) {
    *out += '\n';
    out->append(options.subsequent_indent, ' ');
  }
  *out += ']';
}

}  // namespace

void PrettyPrintJsonAsPython(std::string* out, const ::nlohmann::json& j,
                             const PrettyPrintJsonAsPythonOptions& options) {
  std::size_t existing_size = out->size();
  FormatAsSingleLineForPython(out, j);
  std::ptrdiff_t added_size = out->size() - existing_size;
  int max_width = options.width - options.cur_line_indent;
  if (added_size > max_width) {
    if (const auto* obj = j.get_ptr<const ::nlohmann::json::object_t*>()) {
      out->resize(existing_size);
      PrettyPrintJsonObjectAsPythonInternal(out, *obj, options);
      return;
    } else if (const auto* arr =
                   j.get_ptr<const ::nlohmann::json::array_t*>()) {
      out->resize(existing_size);
      PrettyPrintJsonArrayAsPythonInternal(out, *arr, options);
      return;
    }
  }
}

std::string PrettyPrintJsonAsPython(
    const ::nlohmann::json& j, const PrettyPrintJsonAsPythonOptions& options) {
  std::string out;
  PrettyPrintJsonAsPython(&out, j, options);
  return out;
}

std::string PrettyPrintJsonAsPythonRepr(
    const Result<::nlohmann::json>& j, std::string_view prefix,
    std::string_view suffix, const PrettyPrintJsonAsPythonOptions& options) {
  std::string pretty{prefix};
  const char* dotdotdot = "...";
  if (j.ok()) {
    PrettyPrintJsonAsPythonOptions adjusted_options = options;
    adjusted_options.width -= suffix.size();
    adjusted_options.cur_line_indent += prefix.size();
    PrettyPrintJsonAsPython(&pretty, *j, options);
    dotdotdot = "";
  }
  StrAppend(&pretty, dotdotdot, suffix);
  return pretty;
}

}  // namespace internal_python
}  // namespace tensorstore
