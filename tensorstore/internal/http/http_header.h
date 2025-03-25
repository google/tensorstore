// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_HTTP_HTTP_HEADER_H_
#define TENSORSTORE_INTERNAL_HTTP_HTTP_HEADER_H_

#include <stddef.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {

/// A case-insensitive map of HTTP headers.
class HeaderMap {
 public:
  HeaderMap() = default;
  HeaderMap(const HeaderMap&) = default;
  HeaderMap& operator=(const HeaderMap&) = default;
  HeaderMap(HeaderMap&&) = default;
  HeaderMap& operator=(HeaderMap&&) = default;

  explicit HeaderMap(
      std::initializer_list<std::pair<std::string_view, std::string_view>> il)
      : headers_(il.begin(), il.end()) {}

  using const_iterator =
      absl::btree_map<std::string, std::string>::const_iterator;

  using value_type = absl::btree_map<std::string, std::string>::value_type;

  const_iterator begin() const { return headers_.begin(); }
  const_iterator end() const { return headers_.end(); }

  bool empty() const { return headers_.empty(); }
  size_t size() const { return headers_.size(); }

  template <typename T>
  const_iterator find(T&& key) const {
    return headers_.find(std::forward<T>(key));
  }

  /// Sets a header in `headers`. Overwrites existing headers.
  void SetHeader(std::string_view field_name, std::string_view field_value);
  void SetHeader(std::string_view field_name, std::string field_value);
  void SetHeader(std::string_view field_name, const char* field_value) {
    SetHeader(field_name, std::string_view(field_value));
  }

  /// Sets a header in `headers` and combines duplicate headers using
  /// https://www.rfc-editor.org/rfc/rfc7230.
  void CombineHeader(std::string_view field_name, std::string_view field_value);

  /// Clears a header in `headers`.
  void ClearHeader(std::string_view field_name);

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const HeaderMap& headers) {
    absl::Format(&sink, "<");
    const char* sep = "";
    for (const auto& kv : headers.headers_) {
      sink.Append(sep);
      sink.Append(kv.first);
      sink.Append(": ");
#ifndef NDEBUG
      // Redact auth_token in response logging.
      if (absl::StrContainsIgnoreCase(kv.first, "auth_token")) {
        sink.Append("#####");
      } else if (absl::StrContainsIgnoreCase(kv.first, "authorization")) {
        sink.Append("#####");
      } else
#endif
      {
        sink.Append(kv.second);
      }
      sep = "  ";
    }
    sink.Append(">}");
  }

  /// Attempts to parse a header using SimpleAtoi.
  template <typename T>
  std::optional<T> TryParseIntHeader(std::string_view field_name) const {
    static_assert(std::is_integral_v<T>);
    auto it = headers_.find(field_name);
    if (it != headers_.end()) {
      T result;
      if (absl::SimpleAtoi(it->second, &result)) {
        return result;
      }
    }
    return std::nullopt;
  }

  /// Attempts to parse a header using SimpleAtob.
  std::optional<bool> TryParseBoolHeader(std::string_view field_name) const {
    auto it = headers_.find(field_name);
    if (it != headers_.end()) {
      bool result;
      if (absl::SimpleAtob(it->second, &result)) {
        return result;
      }
    }
    return std::nullopt;
  }

 private:
  absl::btree_map<std::string, std::string> headers_;
};

/// `strptime`-compatible format string for the HTTP date header.
///
/// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Date
///
/// Note that the time zone is always UTC and is specified as "GMT".
constexpr const char kHttpTimeFormat[] = "%a, %d %b %E4Y %H:%M:%S GMT";

/// Validates that `header` is a valid HTTP header line (excluding the final
/// "\r\n").
///
/// The syntax is defined by:
/// https://datatracker.ietf.org/doc/html/rfc7230#section-3.2
///
///   header-field   = field-name ":" OWS field-value OWS
///   field-name     = token
///   field-value    = *( field-content / obs-fold )
///   field-content  = field-vchar [ 1*( SP / HTAB ) field-vchar ]
///   field-vchar    = VCHAR / obs-text
///   OWS            = *( SP / HTAB )
///   VCHAR          = any visible ascii character
///   obs-text       = %x80-FF
///   obs-fold = CRLF 1*( SP / HTAB )
///
/// This function differs from the spec in that it does not accept `obs-fold`
/// (obsolete line folding) syntax.
Result<std::pair<std::string_view, std::string_view>> ValidateHttpHeader(
    std::string_view field_name, std::string_view field_value);
Result<std::pair<std::string_view, std::string_view>> ValidateHttpHeader(
    std::string_view header);

/// Parses `data` as a header block and calls `set_header` for each header
/// field. Returns the number of bytes consumed from `data`.
size_t ParseAndSetHeaders(std::string_view data,
                          absl::FunctionRef<void(std::string_view field_name,
                                                 std::string_view field_value)>
                              set_header);

/// Parses the "content-range" header, which can be used to determine the
/// portion of an object returned by an HTTP request (with status code 206).
/// Returned tuple fields are {start, end, total_length}
std::optional<std::tuple<size_t, size_t, size_t>> TryParseContentRangeHeader(
    const HeaderMap& headers);

/// Try to get the content length from the headers.
std::optional<size_t> TryGetContentLength(const HeaderMap& headers);

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_HTTP_HEADER_H_
