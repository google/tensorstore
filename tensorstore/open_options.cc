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

#include "tensorstore/open_options.h"

#include <utility>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"

namespace tensorstore {

absl::Status SpecOptions::Set(kvstore::Spec value) {
  if (value.valid()) {
    if (kvstore.valid()) {
      return absl::InvalidArgumentError("KvStore already specified");
    }
    kvstore = std::move(value);
  }
  return absl::OkStatus();
}

absl::Status SpecConvertOptions::Set(Context value) {
  if (value && !context) {
    // No error if `context` is already set, because binding a context after a
    // context has already been bound is just a no op.
    context = std::move(value);
  }
  return absl::OkStatus();
}

absl::Status TransactionalOpenOptions::Set(Transaction value) {
  if (value != no_transaction) {
    if (transaction != no_transaction && transaction != value) {
      return absl::InvalidArgumentError("Inconsistent transactions specified");
    }
    transaction = std::move(value);
  }
  return absl::OkStatus();
}

absl::Status TransactionalOpenOptions::Set(KvStore value) {
  if (value.transaction != no_transaction) {
    if (transaction != no_transaction && transaction != value.transaction) {
      return absl::InvalidArgumentError("Inconsistent transactions specified");
    }
    transaction = std::move(value.transaction);
  }
  if (value.valid()) {
    if (kvstore.valid()) {
      return absl::InvalidArgumentError("KvStore already specified");
    }
    kvstore.path = value.path;
    kvstore.driver =
        internal_kvstore::WrapDriverAsDriverSpec(std::move(value.driver));
  }
  return absl::OkStatus();
}

}  // namespace tensorstore
