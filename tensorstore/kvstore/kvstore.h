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
#include "tensorstore/transaction.h"
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

/// Combines a `DriverPtr` with a string path that serves as a key prefix, and
/// an optional transaction.
class KvStore : public KvStorePathBase<DriverPtr> {
 public:
  KvStore() = default;

  KvStore(DriverPtr driver) : KvStorePathBase<DriverPtr>(std::move(driver)) {}

  explicit KvStore(DriverPtr driver, Transaction transaction = no_transaction)
      : KvStorePathBase<DriverPtr>(std::move(driver)),
        transaction(std::move(transaction)) {}

  explicit KvStore(DriverPtr driver, std::string path,
                   Transaction transaction = no_transaction)
      : KvStorePathBase<DriverPtr>(std::move(driver), std::move(path)),
        transaction(std::move(transaction)) {}

  /// Returns a Spec that can be used to re-open this `KvStore`.
  ///
  /// Options that modify the returned `Spec` may be specified in any order.
  /// Refer to `kvstore::Driver::spec` for details on supported options.
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

  /// Returns the URL representation if available.
  Result<std::string> ToUrl() const;

  friend bool operator==(const KvStore& a, const KvStore& b);
  friend bool operator!=(const KvStore& a, const KvStore& b) {
    return !(a == b);
  }

  /// Returns the corresponding non-transactional `KvStore`.
  KvStore non_transactional() const& { return KvStore(driver, path); }

  KvStore non_transactional() && {
    return KvStore(std::move(driver), std::move(path));
  }

  /// Changes to a new transaction.
  ///
  /// Fails if `store` is already associated with an uncommitted transaction.
  ///
  /// This is intended to be used with the "pipeline" `operator|` or
  /// `ChainResult`.
  ///
  /// Example:
  ///
  ///     tensorstore::KvStore store = ...;
  ///     auto transaction = tensorstore::Transaction(tensorstore::isolated);
  ///     TENSORSTORE_ASSIGN_OR_RETURN(store, store | transaction);
  ///
  friend Result<KvStore> ApplyTensorStoreTransaction(KvStore store,
                                                     Transaction transaction) {
    TENSORSTORE_RETURN_IF_ERROR(
        internal::ChangeTransaction(store.transaction, std::move(transaction)));
    return store;
  }

  /// "Pipeline" operator.
  ///
  /// In the expression  `x | y`, if
  ///   * y is a function having signature `Result<U>(T)`
  ///
  /// Then operator| applies y to the value of x, returning a
  /// Result<U>. See tensorstore::Result operator| for examples.
  template <typename Func>
  PipelineResultType<const KvStore&, Func> operator|(Func&& func) const& {
    return std::forward<Func>(func)(*this);
  }
  template <typename Func>
  PipelineResultType<KvStore&&, Func> operator|(Func&& func) && {
    return std::forward<Func>(func)(std::move(*this));
  }

  /// Bound transaction to use for I/O.
  Transaction transaction = no_transaction;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.driver, x.path, x.transaction);
  };
};

/// Driver-agnostic options that may be specified when opening a `Driver`.
/// Refer to the documentation of `kvstore::Open` for details.
struct DriverOpenOptions {
  Context context;

  template <typename T>
  constexpr static bool IsOption = false;

  void Set(Context value) { context = std::move(value); }
};

/// Driver-agnostic options that may be specified when opening a `KvStore`.
/// Refer to the documentation of `kvstore::Open` for details.
struct OpenOptions : public DriverOpenOptions {
  Transaction transaction = no_transaction;

  template <typename T>
  constexpr static bool IsOption = DriverOpenOptions::IsOption<T>;

  using DriverOpenOptions::Set;

  void Set(Transaction transaction) {
    this->transaction = std::move(transaction);
  }
};

template <>
constexpr inline bool DriverOpenOptions::IsOption<Context> = true;
template <>
constexpr inline bool OpenOptions::IsOption<Transaction> = true;

/// Opens a `KvStore` based on an already-parsed `Spec`.
///
/// \param spec KvStore specification.
/// \param options Options for opening the spec.
Future<KvStore> Open(Spec spec, OpenOptions&& options);

/// Same as above, but first parses `json_spec` (which may also be a URL) into a
/// `Spec`.
Future<KvStore> Open(::nlohmann::json json_spec, OpenOptions&& options);

/// Opens a `KvStore` based on an already-parsed `kvstore::Spec` and an optional
/// sequence of options.
///
/// Options may be specified in any order, and are identified by their type.
/// Supported option types are:
///
/// - Context: specifies the context in which to obtain any unbound context
///   resources in `spec`.  Any already-bound context resources in `spec`
///   remain unmodified.  If not specified, `Context::Default()` is used.
///
/// - Transaction: specifies a transaction to bind to the returned `KvStore`.
///   Currently this does not affect the open operation itself.
///
/// \param spec Driver specification.
/// \param option Any option compatible with `OpenOptions`.
template <typename... Option>
static std::enable_if_t<IsCompatibleOptionSequence<OpenOptions, Option...>,
                        Future<KvStore>>
Open(Spec spec, Option&&... option) {
  OpenOptions options;
  (options.Set(option), ...);
  return kvstore::Open(std::move(spec), std::move(options));
}

/// Same as above, but first parses the `Spec` from JSON or URL.
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
