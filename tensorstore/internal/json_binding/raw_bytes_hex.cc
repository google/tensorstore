// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/json_binding/raw_bytes_hex.h"

#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"

namespace tensorstore {
namespace internal_json_binding {

namespace {
bool IsHexString(std::string_view s) {
  for (char c : s) {
    if (!(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f') &&
        !(c >= 'A' && c <= 'F')) {
      return false;
    }
  }
  return true;
}
}  // namespace

namespace raw_bytes_hex_binder {
absl::Status RawBytesHexImpl::operator()(std::true_type is_loading, NoOptions,
                                         void* obj, ::nlohmann::json* j) const {
  auto* s = j->get_ptr<const std::string*>();
  if (!s || s->size() != 2 * num_bytes ||
      !internal_json_binding::IsHexString(*s)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected string with %d hex digits, but received: %s",
                        num_bytes * 2, j->dump()));
  }
  std::string temp = absl::HexStringToBytes(*s);
  assert(temp.size() == num_bytes);
  std::memcpy(obj, temp.data(), num_bytes);
  return absl::OkStatus();
}

absl::Status RawBytesHexImpl::operator()(std::false_type is_loading, NoOptions,
                                         const void* obj,
                                         ::nlohmann::json* j) const {
  *j = absl::BytesToHexString(
      absl::string_view(reinterpret_cast<const char*>(obj), num_bytes));
  return absl::OkStatus();
}

}  // namespace raw_bytes_hex_binder
}  // namespace internal_json_binding
}  // namespace tensorstore
