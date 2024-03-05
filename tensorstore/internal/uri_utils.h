
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

#ifndef TENSORSTORE_INTERNAL_URI_UTILS_H_
#define TENSORSTORE_INTERNAL_URI_UTILS_H_

#include <cstdint>
#include <string>
#include <string_view>

namespace tensorstore {
namespace internal {

/// Set of ASCII characters (0-127) represented as a bit vector.
class AsciiSet {
 public:
  /// Constructs an empty set.
  constexpr AsciiSet() : bitvec_{0, 0} {}

  /// Constructs a set of the characters in `s`.
  constexpr AsciiSet(std::string_view s) : bitvec_{0, 0} {
    for (char c : s) {
      Set(c);
    }
  }

  /// Adds a character to the set.
  constexpr void Set(char c) {
    auto uc = static_cast<unsigned char>(c);
    bitvec_[(uc & 64) ? 1 : 0] |= static_cast<uint64_t>(1) << (uc & 63);
  }

  /// Returns `true` if `c` is in the set.
  constexpr bool Test(char c) const {
    auto uc = static_cast<unsigned char>(c);
    if (uc >= 128) return false;
    return (bitvec_[(uc & 64) ? 1 : 0] >> (uc & 63)) & 1;
  }

 private:
  uint64_t bitvec_[2];
};

static inline constexpr AsciiSet kUriUnreservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_.!~*'()"};

static inline constexpr AsciiSet kUriPathUnreservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-_.!~*'():@&=+$,;/"};

/// Percent encodes any characters in `src` that are not in `unreserved`.
void PercentEncodeReserved(std::string_view src, std::string& dest,
                           AsciiSet unreserved);
inline std::string PercentEncodeReserved(std::string_view src,
                                         AsciiSet unreserved) {
  std::string dest;
  PercentEncodeReserved(src, dest, unreserved);
  return dest;
}

/// Percent-encodes characters not allowed in the URI path component, as defined
/// by RFC2396:
///
/// https://datatracker.ietf.org/doc/html/rfc2396
///
/// Allowed characters are:
///
/// - Unreserved characters: `unreserved` as defined by RFC2396
///   https://datatracker.ietf.org/doc/html/rfc2396#section-2.3
///   a-z, A-Z, 0-9, "-", "_", ".", "!", "~", "*", "'", "(", ")"
///
/// - Path characters: `pchar` as defined by RFC2396
///   https://datatracker.ietf.org/doc/html/rfc2396#section-3.3
///   ":", "@", "&", "=", "+", "$", ","
///
/// - Path segment parameter separator:
///   https://datatracker.ietf.org/doc/html/rfc2396#section-3.3
///   ";"
///
/// - Path segment separator:
///   https://datatracker.ietf.org/doc/html/rfc2396#section-3.3
///   "/"
inline std::string PercentEncodeUriPath(std::string_view src) {
  return PercentEncodeReserved(src, kUriPathUnreservedChars);
}

/// Percent-encodes characters not in the unreserved set, as defined by RFC2396:
///
/// Allowed characters are:
///
/// - Unreserved characters: `unreserved` as defined by RFC2396
///   https://datatracker.ietf.org/doc/html/rfc2396#section-2.3
///   a-z, A-Z, 0-9, "-", "_", ".", "!", "~", "*", "'", "(", ")"
///
/// This is equivalent to the ECMAScript `encodeURIComponent` function:
/// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/encodeURIComponent
inline std::string PercentEncodeUriComponent(std::string_view src) {
  return PercentEncodeReserved(src, kUriUnreservedChars);
}

/// Decodes "%XY" sequences in `src`, where `X` and `Y` are hex digits, to the
/// corresponding character `\xXY`.  "%" characters not followed by 2 hex digits
/// are left unchanged.
///
/// Assigns the decoded result to `dest`.
void PercentDecodeAppend(std::string_view src, std::string& dest);

inline std::string PercentDecode(std::string_view src) {
  std::string dest;
  PercentDecodeAppend(src, dest);
  return dest;
}

struct ParsedGenericUri {
  /// Portion of URI before the initial "://", or empty if there is no "://".
  std::string_view scheme;
  /// Portion of URI after the initial "://" (or from the beginning if there is
  /// no "://") and before the first `?` or `#`.  Not percent decoded.
  std::string_view authority_and_path;
  /// Portion of URI after the first `?` but before the first `#`.  Not percent
  /// decoded.
  std::string_view query;
  /// Portion of URI after the first `#`.  Not percent decoded.
  std::string_view fragment;
};

/// Parses a "generic" URI of the form
/// `<scheme>://<authority-and-path>?<query>#<fragment>` where the `?<query>`
/// and `#<fragment>` portions are optional.
ParsedGenericUri ParseGenericUri(std::string_view uri);

/// Parses the hostname from "authority_and_path".
std::string_view ParseHostname(std::string_view authority_and_path);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_URI_UTILS_H_
