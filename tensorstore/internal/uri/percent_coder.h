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

#ifndef TENSORSTORE_INTERNAL_URI_PERCENT_CODER_H_
#define TENSORSTORE_INTERNAL_URI_PERCENT_CODER_H_

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/internal/uri/ascii_set.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_uri {

// https://datatracker.ietf.org/doc/html/rfc3986#section-2.3
static inline constexpr AsciiSet kUnreserved{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-._~"};

// https://datatracker.ietf.org/doc/html/rfc3986#section-2.2
static inline constexpr AsciiSet kGenDelims{":/?#[]@"};
static inline constexpr AsciiSet kSubDelims{"!$&'()*+,;="};
static inline constexpr AsciiSet kReserved = kGenDelims | kSubDelims;

// https://datatracker.ietf.org/doc/html/rfc3986#section-3.3
static inline constexpr AsciiSet kUriPathUnreservedChars =
    kUnreserved | kSubDelims | AsciiSet{":@/"};

// Same as kUriPathUnreservedChars except that "@" is excluded.
//
// This is used when encoding kvstore URLs in order to reserve @ for specifying
// versions, e.g. for OCDBT.
static inline constexpr AsciiSet kKvStoreUriPathUnreservedChars =
    kUnreserved | kSubDelims | AsciiSet{":/"};

// Decodes a single percent-encoded character, if legal.
std::optional<unsigned char> PercentDecodeChar(std::string_view str);

// Decodes percent-encoded sequences in `src`.
//
// Any `%`-escaped sequence is decoded if the resulting character is in the
// `to_decode` set, otherwise it is left as-is. Characters in `src` that are not
// part of a `%`-escaped sequence are included in the result as is.
//
// Returns an error if `src` contains an invalid `%`-escaped sequence.
absl::Status PercentDecodeAppend(std::string_view src, std::string& dest);
inline Result<std::string> PercentDecode(std::string_view src) {
  std::string dest;
  auto status = PercentDecodeAppend(src, dest);
  if (!status.ok()) return status;
  return dest;
}

// Percent-encodes characters in `src`.
//
// Characters in `src` that are present in the `reserved` set are
// included in the result as is, all other characters converted to their
// percent-encoded (`%xx`) representation.
void PercentEncode(std::string_view src, AsciiSet reserved, std::string& dest);
inline std::string PercentEncode(std::string_view src, AsciiSet reserved) {
  std::string dest;
  PercentEncode(src, reserved, dest);
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
  return PercentEncode(src, kUriPathUnreservedChars);
}

/// Percent-encodes characters not allowed in KvStore URI paths.
///
/// "@" is percent-encoded in order to allow it to be used to indicate a
/// version.
inline std::string PercentEncodeKvStoreUriPath(std::string_view src) {
  return PercentEncode(src, kKvStoreUriPathUnreservedChars);
}

/// Percent-encodes characters not in the unreserved set.
///
/// This is equivalent to the ECMAScript `encodeURIComponent` function.
inline std::string PercentEncodeUriComponent(std::string_view src) {
  return PercentEncode(src, kUnreserved);
}

/// Splits a percent-encoded string into two parts such the right-most part has
/// no percent-encoded delimiters.
std::pair<std::string_view, std::string_view> RSplitPercentEncoded(
    std::string_view src, AsciiSet delimiters);

}  // namespace internal_uri
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_URI_PERCENT_CODER_H_
