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

#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/flags/parse.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"

ABSL_FLAG(tensorstore::JsonAbslFlag<std::optional<tensorstore::kvstore::Spec>>,
          source, std::nullopt, "Source kvstore");
ABSL_FLAG(tensorstore::JsonAbslFlag<std::optional<tensorstore::kvstore::Spec>>,
          target, std::nullopt, "Target kvstore");

namespace tensorstore {

namespace {

Result<int> RunCopy() {
  static absl::Mutex log_mutex;

  auto source_spec = absl::GetFlag(FLAGS_source).value;
  if (!source_spec) {
    return absl::InvalidArgumentError("Must specify --source");
  }
  auto target_spec = absl::GetFlag(FLAGS_target).value;
  if (!target_spec) {
    return absl::InvalidArgumentError("Must specify --target");
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto source,
                               kvstore::Open(*source_spec).result());
  TENSORSTORE_ASSIGN_OR_RETURN(auto target,
                               kvstore::Open(*target_spec).result());

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
            std::cout << "Error reading: " << tensorstore::QuoteString(key)
                      << ": " << read_result.status() << std::endl;
          }
          if (!read_result->has_value()) {
            return absl::OkStatus();
          }
          {
            absl::MutexLock lock(&log_mutex);
            std::cout << "Read: " << tensorstore::QuoteString(key) << std::endl;
          }
          return MapFuture(
              InlineExecutor{},
              [&](const Result<TimestampedStorageGeneration>& stamp)
                  -> Result<void> {
                if (!stamp.ok()) {
                  absl::MutexLock lock(&log_mutex);
                  std::cout
                      << "Error writing: " << tensorstore::QuoteString(key)
                      << ": " << stamp.status() << std::endl;
                  return stamp.status();
                }
                {
                  absl::MutexLock lock(&log_mutex);
                  std::cout << "Wrote: " << tensorstore::QuoteString(key)
                            << std::endl;
                }
                return absl::OkStatus();
              },
              kvstore::Write(target, key, read_result->value));
        },
        kvstore::Read(source, key)));
  }

  bool all_ok = true;
  for (const auto& future : write_futures) {
    const auto& result = future.result();
    if (!result.ok()) all_ok = false;
  }

  return all_ok ? 0 : 1;
}
}  // namespace
}  // namespace tensorstore

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);  // InitTensorstore
  auto result = tensorstore::RunCopy();
  if (!result.ok()) {
    std::cerr << result.status() << std::endl;

    if (absl::IsInvalidArgument(result.status())) {
      std::cerr
          << "Usage: " << argv[0]
          << " --source <source-kvstore-spec> --target <target-kvstore-spec>"
          << std::endl;
    }
    return 1;
  }
  return *result;
}
