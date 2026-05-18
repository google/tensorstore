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

#ifndef TENSORSTORE_TSCLI_OCDBT_CHECK_COMMAND_H_
#define TENSORSTORE_TSCLI_OCDBT_CHECK_COMMAND_H_

#include <stdint.h>

#include <vector>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/tscli/command.h"
#include "tensorstore/tscli/lib/ocdbt_check.h"

namespace tensorstore {
namespace cli {

class OcdbtCheckCommand : public Command {
 public:
  OcdbtCheckCommand();

  absl::Status Run(Context::Spec context_spec) final;

 private:
  std::vector<tensorstore::kvstore::Spec> specs_;
  OcdbtCheckOptions options_;
};

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_OCDBT_CHECK_COMMAND_H_
