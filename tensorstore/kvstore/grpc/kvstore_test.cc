// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/kvstore/kvstore.h"

#include "absl/log/absl_log.h"
#include "tensorstore/internal/init_tensorstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/json_absl_flag.h"

tensorstore::kvstore::Spec DefaultKvStoreSpec() {
  return tensorstore::kvstore::Spec::FromJson(  //
             {
                 {"driver", "memory"},
             })
      .value();
}

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::kvstore::Spec>, kvstore_spec,
          DefaultKvStoreSpec(), "kvstore spec for reading/writing data.");

int main(int argc, char** argv) {
  tensorstore::InitTensorstore(&argc, &argv);

  auto kv = tensorstore::kvstore::Open(absl::GetFlag(FLAGS_kvstore_spec).value)
                .result();
  if (!kv.ok()) {
    ABSL_LOG(INFO) << "Failed to open kvstore: " << kv.status();
    return 2;
  }

  tensorstore::internal::TestKeyValueStoreBasicFunctionality(*kv);
  return 0;
}
