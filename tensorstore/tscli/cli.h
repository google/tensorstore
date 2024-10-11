// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_TSCLI_CLI_H_
#define TENSORSTORE_TSCLI_CLI_H_

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/tscli/args.h"

namespace tensorstore {
namespace cli {

// Kvstore commands
absl::Status RunKvstoreCopy(Context::Spec context_spec, CommandFlags flags);
absl::Status RunKvstoreList(Context::Spec context_spec, CommandFlags flags);

// Tensorstore commands
absl::Status RunTsSearch(Context::Spec context_spec, CommandFlags flags);
absl::Status RunTsPrintSpec(Context::Spec context_spec, CommandFlags flags);
absl::Status RunTsPrintStorageStatistics(Context::Spec context_spec,
                                         CommandFlags flags);
}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_CLI_H_
