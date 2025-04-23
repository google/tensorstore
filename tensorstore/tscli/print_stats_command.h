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

#ifndef TENSORSTORE_TSCLI_PRINT_STATS_COMMAND_H_
#define TENSORSTORE_TSCLI_PRINT_STATS_COMMAND_H_

#include <vector>

#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "tensorstore/spec.h"
#include "tensorstore/tscli/command.h"

namespace tensorstore {
namespace cli {

// Print the storage statistics for a TensorStore.
class PrintStatsCommand : public Command {
 public:
  PrintStatsCommand();

  absl::Status Run(Context::Spec context_spec) override;

 private:
  std::vector<tensorstore::Spec> specs_;
  std::vector<Box<>> boxes_;

  bool brief_ = true;
  bool has_spec_ = false;
};

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_PRINT_STATS_COMMAND_H_
