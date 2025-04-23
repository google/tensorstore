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

#ifndef TENSORSTORE_TSCLI_LIB_TS_PRINT_STATS_H_
#define TENSORSTORE_TSCLI_LIB_TS_PRINT_STATS_H_

#include <ostream>

#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "tensorstore/spec.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace cli {

absl::Status TsPrintStoredChunks(Context context, tensorstore::Spec spec,
                                 bool brief, std::ostream& output);

absl::Status TsPrintStorageStatistics(tensorstore::Context context,
                                      tensorstore::Spec spec,
                                      tensorstore::span<Box<>> boxes,
                                      bool brief, std::ostream& output);

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_LIB_TS_PRINT_STATS_H_
