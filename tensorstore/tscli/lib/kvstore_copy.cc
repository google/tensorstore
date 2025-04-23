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

#include "tensorstore/tscli/lib/kvstore_copy.h"

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace cli {

absl::Status KvstoreCopy(Context context,
                         tensorstore::kvstore::Spec source_spec,
                         tensorstore::kvstore::Spec target_spec,
                         std::ostream& output) {
  static absl::Mutex log_mutex;

  TENSORSTORE_ASSIGN_OR_RETURN(auto source,
                               kvstore::Open(source_spec, context).result());
  TENSORSTORE_ASSIGN_OR_RETURN(auto target,
                               kvstore::Open(target_spec, context).result());

  TENSORSTORE_ASSIGN_OR_RETURN(auto list_entries,
                               kvstore::ListFuture(source).result());

  std::vector<Future<const void>> write_futures;

  for (const auto& entry : list_entries) {
    std::string key = entry.key;
    write_futures.push_back(MapFutureValue(
        InlineExecutor{},
        [&](const Result<kvstore::ReadResult>& read_result) -> Future<void> {
          if (!read_result.ok()) {
            absl::MutexLock lock(&log_mutex);
            output << "Error reading: " << tensorstore::QuoteString(key) << ": "
                   << read_result.status() << std::endl;
          }
          if (!read_result->has_value()) {
            return absl::OkStatus();
          }
          {
            absl::MutexLock lock(&log_mutex);
            output << "Read: " << tensorstore::QuoteString(key) << std::endl;
          }
          return MapFuture(
              InlineExecutor{},
              [&](const Result<TimestampedStorageGeneration>& stamp)
                  -> Result<void> {
                if (!stamp.ok()) {
                  absl::MutexLock lock(&log_mutex);
                  output << "Error writing: " << tensorstore::QuoteString(key)
                         << ": " << stamp.status() << std::endl;
                  return stamp.status();
                }
                {
                  absl::MutexLock lock(&log_mutex);
                  output << "Wrote: " << tensorstore::QuoteString(key)
                         << std::endl;
                }
                return absl::OkStatus();
              },
              kvstore::Write(target, key, read_result->value));
        },
        kvstore::Read(source, key)));
  }

  absl::Status status = absl::OkStatus();
  for (const auto& future : write_futures) {
    const auto& result = future.result();
    status.Update(result.status());
  }
  return status;
}

}  // namespace cli
}  // namespace tensorstore
