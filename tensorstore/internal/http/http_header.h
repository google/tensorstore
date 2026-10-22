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
#include "tensorstore/internal/uri/ascii_set.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_http {

/// RFC 9110 Section 5.6.2 `tchar` character set:
///   tchar = "!" / "#" / "$" / "%" / "&" / "'" / "*" / "+" / "-" / "." /
///           "^" / "_" / "`" / "|" / "~" / DIGIT / ALPHA
inline constexpr internal_uri::AsciiSet kHttpTokenChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "!#$%&'*+-.^_`|~"};

/// Returns true if `s` is a non-empty HTTP `token` (`1*tchar`).
inline constexpr bool IsValidHttpToken(std::string_view s) {
  return !s.empty() && kHttpTokenChars.ContainsAll(s);
}

/// Visible ASCII characters (`VCHAR`, `0x21..0x7E`).
inline constexpr internal_uri::AsciiSet kAsciiVChar{
    "!\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"};

/// Valid ASCII header field-value characters (`HTAB`, `SP`, and `VCHAR`).
inline constexpr internal_uri::AsciiSet kValidAsciiHeaderValueChars =
    kAsciiVChar | internal_uri::AsciiSet{"\t "};

/// Returns true if `s` is a valid HTTP header field value (RFC 9110 §5.5:
/// `HTAB`, `SP`, `VCHAR` `0x21..0x7E`, or `obs-text` `0x80..0xFF`; no `CR`,
/// `LF`, `NUL`, or other control characters).
inline constexpr bool IsValidHttpHeaderValue(std::string_view s) {
  for (char c : s) {
    if (!kValidAsciiHeaderValueChars.Test(c) &&
        static_cast<unsigned char>(c) < 0x80) {
      return false;
    }
  }
  return true;
}

/// Strips leading and trailing RFC 9110 Section 5.6.3 optional whitespace
/// (`OWS = *( SP / HTAB )`).
inline constexpr std::string_view StripHttpOws(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) {
    s.remove_prefix(1);
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
    s.remove_suffix(1);
  }
  return s;
}

/// Returns true if `name` is a credential-bearing HTTP header (`Authorization`,
/// `Proxy-Authorization`, `Cookie`, `Set-Cookie`, `X-Goog-Encryption-Key`,
/// `X-Amz-Server-Side-Encryption-Customer-Key`, `X-Ms-Encryption-Key`, etc.).
bool IsSensitiveHttpHeader(std::string_view name);

/// Returns `"[REDACTED]"` if `IsSensitiveHttpHeader(name)`, else `value`.
std::string_view RedactHeaderValueForLog(std::string_view name,
                                         std::string_view value);

/// Redacts the value portion of a serialized `"Name: Value"` line if `Name`
/// is a sensitive HTTP header.
std::string RedactH1HeaderLineForLog(std::string_view line);

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
      // Redact sensitive headers in logging across all build modes.
      if (IsSensitiveHttpHeader(kv.first)) {
        sink.Append("#####");
      } else {
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
