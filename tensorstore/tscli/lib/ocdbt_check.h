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

#ifndef TENSORSTORE_TSCLI_LIB_OCDBT_CHECK_H_
#define TENSORSTORE_TSCLI_LIB_OCDBT_CHECK_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/spec.h"

namespace tensorstore {
namespace cli {

inline constexpr size_t kOcdbtCheckDefaultConcurrency = 256;

struct OcdbtCheckOptions {
  /// Optional version to check (e.g. "v1", "v2", or timestamp).
  /// If omitted, all versions are checked.
  std::optional<std::string> version;

  /// If true, provides more verbose output about each check.
  bool detailed = false;

  /// Byte alignment used when calculating unused ranges.
  uint64_t alignment = 4096;

  /// Limit on concurrent node reads.
  size_t concurrency = kOcdbtCheckDefaultConcurrency;
};

/// Verifies the structural integrity of an OCDBT database.
///
/// \param context Context to use for opening the kvstore.
/// \param source_spec Spec of the base kvstore containing the OCDBT database.
/// \param output Stream to write error reports and progress to.
/// \param options Optional settings for the check.
/// \return absl::OkStatus() if the check completed (even if errors were found).
///         An error status if the check itself failed to run (e.g. unable to
///         open kvstore).
absl::Status OcdbtCheck(Context context, tensorstore::kvstore::Spec source_spec,
                        std::ostream& output, OcdbtCheckOptions options = {});

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_LIB_OCDBT_CHECK_H_
