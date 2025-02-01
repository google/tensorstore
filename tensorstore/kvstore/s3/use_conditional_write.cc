// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/s3/use_conditional_write.h"

#include <optional>
#include <string_view>

#include "absl/flags/flag.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "re2/re2.h"
#include "tensorstore/internal/env.h"

// NOTE: This flag is used to override the default heuristic for
// determining whether conditional writes should be used.
// It should be considered experimental; it will likely migrate to a spec
// option in the future.
ABSL_FLAG(std::optional<bool>, tensorstore_s3_use_conditional_write,
          std::nullopt,
          "Use S3 conditional operations for write requests."
          "Overrides TENSORSTORE_S3_USE_CONDITIONAL_WRITE");

using ::tensorstore::internal::GetFlagOrEnvValue;

namespace tensorstore {
namespace internal_kvstore_s3 {

bool IsAwsS3Endpoint(std::string_view endpoint) {
  static LazyRE2 kIsAwsS3Endpoint = {
      "(^|[.])"
      "s3(-fips|-accesspoint|-accesspoint-fips)?[.]"
      "(dualstack[.])?"
      "([^.]+[.])?"
      "amazonaws[.]com$"};

  endpoint = absl::StripPrefix(endpoint, "https://");
  endpoint = absl::StripPrefix(endpoint, "http://");
  if (endpoint.empty()) return false;
  return RE2::PartialMatch(*absl::StrSplit(endpoint, '/').begin(),
                           *kIsAwsS3Endpoint);
}

bool UseConditionalWrite(std::string_view endpoint) {
  return GetFlagOrEnvValue(FLAGS_tensorstore_s3_use_conditional_write,
                           "TENSORSTORE_S3_USE_CONDITIONAL_WRITE")
      .value_or(IsAwsS3Endpoint(endpoint));
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
