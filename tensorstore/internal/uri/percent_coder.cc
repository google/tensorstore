// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/uri/percent_coder.h"

#include <stddef.h>

#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/uri/ascii_set.h"
#include "tensorstore/internal/utf8.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_uri {
namespace {

static inline constexpr AsciiSet kHexDigits{"0123456789ABCDEFabcdef"};

inline char IntToHexDigit(int x) {
  assert(x >= 0 && x < 16);
  return "0123456789ABCDEF"[x];
}

}  // namespace

std::optional<unsigned char> PercentDecodeChar(std::string_view str) {
  if (str.size() != 3 || str[0] != '%' || !kHexDigits(str[1]) ||
      !kHexDigits(str[2])) {
    return std::nullopt;
  }
  unsigned int decoded_c;
  if (!absl::SimpleHexAtoi(str.substr(1), &decoded_c)) {
    return std::nullopt;
  }
  return static_cast<unsigned char>(decoded_c);
}

absl::Status PercentDecodeAppend(std::string_view src, std::string& dest) {
  dest.reserve(dest.size() + src.size());
  std::string_view remaining = src;
  while (!remaining.empty()) {
    auto pos = remaining.find('%');
    if (pos == std::string_view::npos) {
      dest.append(remaining);
      break;
    }
    if (pos > 0) {
      dest.append(remaining.substr(0, pos));
      remaining.remove_prefix(pos);
    }
    if (auto c = PercentDecodeChar(remaining.substr(0, 3)); c.has_value()) {
      dest.push_back(*c);
      remaining.remove_prefix(3);
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid percent-encoding at position %d: %v",
                          src.data() - remaining.data(), QuoteString(src)));
    }
  }

  // Require the output to be valid UTF-8.
  if (!internal::IsValidUtf8(dest)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid UTF-8 percent-encoding sequence: %v", QuoteString(src)));
  }
  return absl::OkStatus();
}

void PercentEncode(std::string_view src, AsciiSet reserved, std::string& dest) {
  size_t num_encoded = 0;
  for (char c : src) {
    if (!reserved.Test(c)) ++num_encoded;
  }
  if (num_encoded == 0) {
    dest.append(src);
    return;
  }
  dest.reserve(dest.size() + src.size() + 2 * num_encoded);
  char buffer[4] = {'%', 0, 0, 0};
  for (char c : src) {
    if (!reserved.Test(c)) {
      buffer[1] = IntToHexDigit(static_cast<unsigned char>(c) / 16);
      buffer[2] = IntToHexDigit(static_cast<unsigned char>(c) % 16);
      dest.append(buffer, 3);
    } else {
      dest.push_back(c);
    }
  }
}

std::pair<std::string_view, std::string_view> RSplitPercentEncoded(
    std::string_view src, AsciiSet delimiters) {
  size_t pos = src.size();
  while (pos > 0) {
    size_t percent_pos = src.rfind('%', pos - 1);
    if (percent_pos == std::string_view::npos) {
      break;
    }
    std::string_view encoded_char = src.substr(percent_pos, 3);
    if (auto c = PercentDecodeChar(encoded_char); c.has_value()) {
      if (delimiters.Test(*c)) {
        return {src.substr(0, percent_pos + 3), src.substr(percent_pos + 3)};
      }
    }
    pos = percent_pos;
  }
  return {src.substr(0, 0), src};
}

}  // namespace internal_uri
}  // namespace tensorstore
