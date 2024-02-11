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

#ifndef TENSORSTORE_KVSTORE_S3_URI_UTILS_H
#define TENSORSTORE_KVSTORE_S3_URI_UTILS_H

#include <string>
#include <string_view>

#include "tensorstore/internal/uri_utils.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

// See description of function UriEncode at this URL
// https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
static inline constexpr internal::AsciiSet kUriUnreservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-._~"};

// NOTE: Only adds "/" to kUriUnreservedChars
static inline constexpr internal::AsciiSet kUriKeyUnreservedChars{
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "/-._~"};

inline std::string S3UriEncode(std::string_view src) {
  std::string dest;
  internal::PercentEncodeReserved(src, dest, kUriUnreservedChars);
  return dest;
}

inline std::string S3UriObjectKeyEncode(std::string_view src) {
  std::string dest;
  internal::PercentEncodeReserved(src, dest, kUriKeyUnreservedChars);
  return dest;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_URI_UTILS_H
