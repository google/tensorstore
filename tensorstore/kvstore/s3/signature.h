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

#ifndef TENSORSTORE_KVSTORE_S3_SIGNATURE_H_
#define TENSORSTORE_KVSTORE_S3_SIGNATURE_H_


/// \file
/// S3 Authorization Signature calculations.

#include <map>
#include <string>
#include <string_view>

#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"

using ::tensorstore::internal::ParsedGenericUri;

namespace tensorstore {
namespace internal_storage_s3 {


Result<std::string> CanonicalRequest(
    std::string_view http_method,
    const ParsedGenericUri & uri,
    const std::map<std::string, std::string> & headers,
    std::string_view payload_hash);

std::string SigningString(
    std::string_view canonical_request,
    const absl::Time & time,
    std::string_view aws_region);

std::string Signature(
    std::string_view aws_secret_access_key,
    std::string_view aws_region,
    const absl::Time & time,
    std::string_view signing_string);

} // namespace tensorstore
} // namespace internal_storage_s3

#endif // TENSORSTORE_KVSTORE_S3_SIGNATURE_H_
