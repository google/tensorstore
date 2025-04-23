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

#ifndef TENSORSTORE_TSCLI_SEARCH_COMMAND_H_
#define TENSORSTORE_TSCLI_SEARCH_COMMAND_H_

#include <vector>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/tscli/command.h"

namespace tensorstore {
namespace cli {

// Search for Tensorstores under a given kvstore spec.
class SearchCommand : public Command {
 public:
  SearchCommand();

  absl::Status Run(Context::Spec context_spec) override;

 private:
  std::vector<tensorstore::kvstore::Spec> specs_;
  bool brief_ = true;
};

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_SEARCH_COMMAND_H_
