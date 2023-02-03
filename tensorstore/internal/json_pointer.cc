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

#include "tensorstore/internal/json_pointer.h"

#include <algorithm>
#include <string_view>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include <nlohmann/json.hpp>
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace json_pointer {

absl::Status Validate(std::string_view s) {
  // From RFC 6901:
  //
  // A JSON Pointer is a Unicode string (see [RFC4627], Section 3)
  // containing a sequence of zero or more reference tokens, each prefixed
  // by a '/' (%x2F) character.
  //
  // Because the characters '~' (%x7E) and '/' (%x2F) have special
  // meanings in JSON Pointer, '~' needs to be encoded as '~0' and '/'
  // needs to be encoded as '~1' when these characters appear in a
  // reference token.
  //
  // The ABNF syntax of a JSON Pointer is:
  //
  //    json-pointer    = *( "/" reference-token )
  //    reference-token = *( unescaped / escaped )
  //    unescaped       = %x00-2E / %x30-7D / %x7F-10FFFF
  //       ; %x2F ('/') and %x7E ('~') are excluded from 'unescaped'
  //    escaped         = "~" ( "0" / "1" )
  //      ; representing '~' and '/', respectively
  if (s.empty()) {
    return absl::OkStatus();
  }

  const auto parse_error = [&](const auto&... message) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat(message..., ": ", tensorstore::QuoteString(s)));
  };
  if (s[0] != '/') {
    return parse_error("JSON Pointer does not start with '/'");
  }
  for (size_t i = 1; i < s.size(); ++i) {
    if (s[i] != '~') continue;
    if (i + 1 == s.size() || (s[i + 1] != '0' && s[i + 1] != '1')) {
      return parse_error(
          "JSON Pointer requires '~' to be followed by '0' or '1'");
    }
    ++i;
  }
  return absl::OkStatus();
}

namespace {
unsigned char DecodeEscape(char x) {
  assert(x == '0' || x == '1');
  return x == '0' ? '~' : '/';
}

void DecodeReferenceToken(std::string_view encoded_token, std::string& output) {
  output.clear();
  output.reserve(encoded_token.size());
  for (size_t i = 0; i < encoded_token.size(); ++i) {
    char c = encoded_token[i];
    switch (c) {
      case '~':
        ++i;
        assert(i != encoded_token.size());
        output += DecodeEscape(encoded_token[i]);
        break;
      default:
        output += c;
    }
  }
}
}  // namespace

CompareResult Compare(std::string_view a, std::string_view b) {
  const size_t mismatch_index = std::distance(
      a.begin(), std::mismatch(a.begin(), a.end(), b.begin(), b.end()).first);
  if (mismatch_index == a.size()) {
    if (mismatch_index == b.size()) return kEqual;
    if (b[mismatch_index] == '/') {
      // a = "X"
      // b = "X/Y"
      return kContains;
    }
    // a = "X"
    // b = "XY"
    return kLessThan;
  }
  if (mismatch_index == b.size()) {
    if (a[mismatch_index] == '/') {
      // a = "X/Y"
      // b = "X"
      return kContainedIn;
    }
    // a = "XY"
    // b = "X"
    return kGreaterThan;
  }
  if (a[mismatch_index] == '/') {
    // a = "X/Y"
    // b = "XZ"
    return kLessThan;
  }
  if (b[mismatch_index] == '/') {
    // a = "XZ"
    // b = "X/Y"
    return kGreaterThan;
  }
  // Need to compare final character.
  unsigned char a_char, b_char;
  if (a[mismatch_index - 1] == '~') {
    // Mismatch must not be in first character, since non-empty JSON Pointer
    // must start with '/'.
    assert(mismatch_index > 0);
    // a = "X~IY"
    // b = "X~JZ"
    a_char = DecodeEscape(a[mismatch_index]);
    b_char = DecodeEscape(b[mismatch_index]);
  } else {
    if (a[mismatch_index] == '~') {
      // a = "X~IY"
      // b = "XZ"
      assert(mismatch_index + 1 < a.size());
      a_char = DecodeEscape(a[mismatch_index + 1]);
    } else {
      a_char = a[mismatch_index];
    }
    if (b[mismatch_index] == '~') {
      // a = "XY"
      // b = "X~IZ"
      assert(mismatch_index + 1 < b.size());
      b_char = DecodeEscape(b[mismatch_index + 1]);
    } else {
      b_char = b[mismatch_index];
    }
  }
  return a_char < b_char ? kLessThan : kGreaterThan;
}

std::string EncodeReferenceToken(std::string_view token) {
  std::string result;
  result.reserve(token.size());
  for (char c : token) {
    switch (c) {
      case '~':
        result += {'~', '0'};
        break;
      case '/':
        result += {'~', '1'};
        break;
      default:
        result += c;
    }
  }
  return result;
}

Result<::nlohmann::json*> Dereference(::nlohmann::json& full_value,
                                      std::string_view sub_value_pointer,
                                      DereferenceMode mode) {
  if (sub_value_pointer.empty()) {
    if (full_value.is_discarded()) {
      if (mode == kMustExist) {
        return absl::NotFoundError("");
      }
      if (mode == kDelete) {
        return nullptr;
      }
    }
    return &full_value;
  }
  assert(sub_value_pointer[0] == '/');
  size_t i = 1;
  auto* sub_value = &full_value;
  std::string decoded_reference_token;
  while (true) {
    if (sub_value->is_discarded()) {
      switch (mode) {
        case kMustExist:
          return absl::NotFoundError("");
        case kCreate:
          *sub_value = ::nlohmann::json::object_t();
          break;
        case kSimulateCreate:
        case kDelete:
          return nullptr;
      }
    }
    size_t pointer_component_end = sub_value_pointer.find('/', i);
    const bool is_leaf = pointer_component_end == std::string_view::npos;
    const auto quoted_pointer = [&] {
      return tensorstore::QuoteString(
          sub_value_pointer.substr(0, pointer_component_end));
    };
    std::string_view pointer_component =
        sub_value_pointer.substr(i, pointer_component_end - i);
    if (auto* j_obj = sub_value->get_ptr<::nlohmann::json::object_t*>()) {
      DecodeReferenceToken(pointer_component, decoded_reference_token);
      if (mode == kCreate) {
        sub_value = &j_obj
                         ->emplace(decoded_reference_token,
                                   ::nlohmann::json::value_t::discarded)
                         .first->second;
      } else if (mode == kDelete && is_leaf) {
        j_obj->erase(decoded_reference_token);
        return nullptr;
      } else {
        auto it = j_obj->find(decoded_reference_token);
        if (it == j_obj->end()) {
          switch (mode) {
            case kSimulateCreate:
            case kDelete:
              return nullptr;
            case kMustExist:
              return absl::NotFoundError(
                  tensorstore::StrCat("JSON Pointer ", quoted_pointer(),
                                      " refers to non-existent object member"));
            case kCreate:
              ABSL_UNREACHABLE();  // COV_NF_LINE
          }
        }
        sub_value = &it->second;
      }
    } else if (auto* j_array =
                   sub_value->get_ptr<::nlohmann::json::array_t*>()) {
      if (pointer_component == "-") {
        switch (mode) {
          case kMustExist:
            return absl::FailedPreconditionError(
                tensorstore::StrCat("JSON Pointer ", quoted_pointer(),
                                    " refers to non-existent array element"));
          case kCreate:
            sub_value =
                &j_array->emplace_back(::nlohmann::json::value_t::discarded);
            break;
          case kSimulateCreate:
          case kDelete:
            return nullptr;
        }
      } else {
        size_t array_index;
        if (pointer_component.empty() ||
            std::any_of(pointer_component.begin(), pointer_component.end(),
                        [](char c) { return !absl::ascii_isdigit(c); }) ||
            (pointer_component.size() > 1 && pointer_component[0] == '0') ||
            !absl::SimpleAtoi(pointer_component, &array_index)) {
          return absl::FailedPreconditionError(
              tensorstore::StrCat("JSON Pointer ", quoted_pointer(),
                                  " is invalid for array value"));
        }
        if (array_index >= j_array->size()) {
          if (mode == kDelete) return nullptr;
          return absl::OutOfRangeError(tensorstore::StrCat(
              "JSON Pointer ", quoted_pointer(),
              " is out-of-range for array of size ", j_array->size()));
        }
        if (mode == kDelete && is_leaf) {
          j_array->erase(j_array->begin() + array_index);
          return nullptr;
        }
        sub_value = &(*j_array)[array_index];
      }
    } else {
      return absl::FailedPreconditionError(tensorstore::StrCat(
          "JSON Pointer reference ", quoted_pointer(), " cannot be applied to ",
          sub_value->type_name(), " value: ", *sub_value));
    }
    if (pointer_component_end == std::string_view::npos) {
      assert(mode != kDelete);
      return sub_value;
    }
    i += pointer_component.size() + 1;
  }
}

Result<const ::nlohmann::json*> Dereference(const ::nlohmann::json& full_value,
                                            std::string_view sub_value_pointer,
                                            DereferenceMode mode) {
  assert(mode == kMustExist || mode == kSimulateCreate);
  return json_pointer::Dereference(const_cast<::nlohmann::json&>(full_value),
                                   sub_value_pointer, mode);
}

absl::Status Replace(::nlohmann::json& full_value,
                     std::string_view sub_value_pointer,
                     ::nlohmann::json new_sub_value) {
  if (sub_value_pointer.empty()) {
    full_value = std::move(new_sub_value);
    return absl::OkStatus();
  }
  if (!new_sub_value.is_discarded()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto* sub_value,
        json_pointer::Dereference(full_value, sub_value_pointer, kCreate));
    *sub_value = std::move(new_sub_value);
    return absl::OkStatus();
  }
  TENSORSTORE_RETURN_IF_ERROR(
      json_pointer::Dereference(full_value, sub_value_pointer, kDelete));
  return absl::OkStatus();
}

}  // namespace json_pointer
}  // namespace tensorstore
