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

#ifndef TENSORSTORE_KVSTORE_KVSTORE_H_
#define TENSORSTORE_KVSTORE_KVSTORE_H_

#include "tensorstore/context.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/option.h"

namespace tensorstore {
namespace kvstore {

/// Options that may be specified when requesting the `Spec` for an open `Path`.
/// Refer to the documentation of `Path::spec` for details.
struct SpecRequestOptions {
  ContextBindingMode context_binding_mode = ContextBindingMode::unspecified;

  template <typename T>
  constexpr static bool IsOption = false;

  void Set(ContextBindingMode value) {
    if (value > context_binding_mode) context_binding_mode = value;
  }
};

template <>
constexpr inline bool SpecRequestOptions::IsOption<ContextBindingMode> = true;

/// Combines a `Driver::Ptr` with a string path that serves as a key prefix.
class KvStore : public KvStorePathBase<DriverPtr> {
 public:
  using KvStorePathBase<DriverPtr>::KvStorePathBase;

  /// Returns a Spec that can be used to re-open this Path.
  ///
  /// Options that modify the returned `Spec::Path` may be specified in any
  /// order.  Refer to `KeyValueStore::spec` for details on supported options.
  ///
  /// \param option Any option compatible with `SpecRequestOptions`.
  /// \error `absl::StatusCode::kUnimplemented` if a JSON representation is
  ///     not supported.  (This behavior is provided by the default
  ///     implementation.)
  template <typename... Option>
  std::enable_if_t<IsCompatibleOptionSequence<SpecRequestOptions, Option...>,
                   Result<Spec>>
  spec(Option&&... option) const {
    SpecRequestOptions options;
    (options.Set(std::move(option)), ...);
    return spec(std::move(options));
  }

  /// Returns a Spec that can be used to re-open this KvStore.
  ///
  /// \param option Options that may modify the returned `Spec`.
  Result<Spec> spec(SpecRequestOptions&& options) const;

  friend bool operator==(const KvStore& a, const KvStore& b);
  friend bool operator!=(const KvStore& a, const KvStore& b) {
    return !(a == b);
  }
};

/// Driver-agnostic options that may be specified when opening a `KvStore`.
/// Refer to the documentation of `kvstore::Open` for details.
struct OpenOptions {
  Context context;

  template <typename T>
  constexpr static bool IsOption = false;

  void Set(Context value) { context = std::move(value); }
};

template <>
constexpr inline bool OpenOptions::IsOption<Context> = true;

/// Opens a `KeyValueStore` based on an already-parsed `Spec`.
///
/// \param spec KeyValueStore path specification.
/// \param options Options for opening the spec.
Future<KvStore> Open(Spec spec, OpenOptions&& options);

/// Same as above, but first parses `json_spec` into a `Spec`.
Future<KvStore> Open(::nlohmann::json json_spec, OpenOptions&& options);

template <typename... Option>
static std::enable_if_t<IsCompatibleOptionSequence<OpenOptions, Option...>,
                        Future<KvStore>>
Open(Spec spec, Option&&... option) {
  OpenOptions options;
  (options.Set(option), ...);
  return kvstore::Open(std::move(spec), std::move(options));
}

/// Same as above, but first parses the `Spec` from JSON.
///
/// \param j JSON specification.
/// \param option Any option compatible with `OpenOptions`.
template <typename... Option>
static std::enable_if_t<IsCompatibleOptionSequence<OpenOptions, Option...>,
                        Future<KvStore>>
Open(::nlohmann::json j, Option&&... option) {
  OpenOptions options;
  (options.Set(option), ...);
  return kvstore::Open(std::move(j), std::move(options));
}

}  // namespace kvstore

using kvstore::KvStore;

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::kvstore::KvStore)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::KvStore)

#endif  // TENSORSTORE_KVSTORE_KVSTORE_H_
