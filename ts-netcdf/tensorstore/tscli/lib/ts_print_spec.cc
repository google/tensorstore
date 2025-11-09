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

#include "tensorstore/tscli/lib/ts_print_spec.h"

#include <iostream>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace cli {

absl::Status TsPrintSpec(Context context, tensorstore::Spec spec,
                         IncludeDefaults include_defaults,
                         std::ostream& output) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto ts,
      tensorstore::Open(spec, context, tensorstore::ReadWriteMode::read,
                        tensorstore::OpenMode::open)
          .result());
  TENSORSTORE_ASSIGN_OR_RETURN(auto actual_spec, ts.spec());
  TENSORSTORE_ASSIGN_OR_RETURN(auto json_spec,
                               actual_spec.ToJson(include_defaults));
  output << json_spec.dump() << std::endl;
  return absl::OkStatus();
}

}  // namespace cli
}  // namespace tensorstore
